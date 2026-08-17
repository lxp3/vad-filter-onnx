#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import contextlib
import os
import sys

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxsim import simplify

opset_version = 18


@contextlib.contextmanager
def patch_bool_eye():
    """
    MossFormer's attention (mossformer.py's MossFormer.cal_attention) builds
    a diagonal mask via `torch.eye(n, dtype=torch.bool)`. ONNX Runtime's
    CPU EyeLike kernel doesn't implement a bool output type, so the exported
    graph fails to load. Patch torch.eye for the duration of export/verify
    to build bool identity matrices via an Equal-of-arange comparison
    instead, which traces to a plain (well-supported) Equal node and is
    numerically identical.
    """
    real_eye = torch.eye

    def patched_eye(n, dtype=None, device=None, **kwargs):
        if dtype == torch.bool:
            idx = torch.arange(n, device=device)
            return idx.unsqueeze(0) == idx.unsqueeze(1)
        return real_eye(n, dtype=dtype, device=device, **kwargs)

    torch.eye = patched_eye
    try:
        yield
    finally:
        torch.eye = real_eye


"""
MossFormerGAN_SE_16K offline denoise export notes
====================================================

MossFormerGAN_SE_16K (alibabasglab/MossFormerGAN_SE_16K) is a GAN-based
speech enhancement model whose generator (SyncANet) is a fully offline,
non-causal spectrogram-in/spectrogram-out network: attention blocks and
FSMN modules run over the whole utterance with no recurrent cache. The
upstream reference pipeline (ClearerVoice-Studio) runs STFT/power-compress
outside the model, feeds the compressed complex spectrogram to SyncANet,
then runs power-uncompress/iSTFT outside the model again. This exporter
bakes all of that (RMS normalization, STFT, power-compress, the generator,
power-uncompress, iSTFT, denormalization) into a single wrapper so the
ONNX graph takes and returns a raw waveform directly, matching this
project's frcrn_se_16k.onnx convention.

Only the generator (SyncANet) checkpoint is used; the GAN discriminator
(last_best_checkpoint.disc.pt) is training-only and not needed for
inference.

Audio format
------------
- Sample rate: 16000 Hz only.
- Input tensor name: speech.
- Input tensor shape: [1, num_samples] (genuinely dynamic — see below).
- Input dtype: float32, normalized float in [-1, 1].
- Output tensor name: enhanced.
- Output tensor shape: [1, num_samples] float32.

Dynamic-length export fix
--------------------------
- SyncANetBlock.forward (generator.py) computes
  `T = math.ceil((old_T - emb_ks) / emb_hs) * emb_hs + emb_ks` from
  old_T = x.shape[2]. math.ceil() is a plain Python builtin, not a torch
  op, so under trace-based ONNX export it collapses that traced dynamic
  size into a concrete Python int (whatever old_T was at trace time)
  instead of an ONNX Shape/Gather sequence; every later `.view([B, T,
  ...])` in that method then bakes the trace-time frame count into the
  graph, so any other input length fails at runtime with a Reshape
  shape-mismatch error.
- For SyncANet, emb_ks=2 and emb_hs=1 always, making that formula the
  identity ceil((n-2)/1)*1+2 == n for any integer n — the padding it
  computes is therefore always exactly zero. `load_mossformergan_se_16k`
  installs an `OnnxSyncANetBlock` override (monkeypatched into the
  generator module before `SyncANet` is constructed, since
  `SyncANet.__init__` looks up `SyncANetBlock` by name from module
  globals) that skips the ceil arithmetic and the resulting no-op F.pad,
  using old_T/old_Q directly — those trace as genuinely dynamic sizes
  since they never pass through a non-tensor Python builtin. Behavior is
  bit-identical to upstream; this only fixes ONNX export to generalize
  across input lengths (verified below at two different lengths, as with
  frcrn_se_16k.onnx).

Model internals
----------------
- STFT: win_len=400 (25 ms), win_inc=100 (6.25 ms hop), fft_len=400,
  periodic Hamming window, complex (real/imag) feature type.
- Upstream applies an RMS normalization (norm_factor = sqrt(T / sum(x^2)))
  before STFT and undoes it after iSTFT; this is baked into the wrapper so
  the graph is still a plain waveform-in/waveform-out model.
- Upstream's power-compress/uncompress (mag ** 0.3 / mag ** (1/0.3)) is
  reimplemented with plain torch ops (torch.complex/abs/angle/cos/sin), all
  ONNX-exportable.
- No recurrent cache: SyncANetBlock's attention + UniDeepFsmn modules are
  feedforward given the whole utterance; there is no caches_in/out I/O.

Streaming
---------
- streaming=0 in ONNX metadata. This model must never be called
  incrementally; the C++ backend (MossformerganSe16kDenoiseModel) buffers
  all input and only invokes the ONNX session once, on input_finished=true.
"""


def get_args():
    parser = argparse.ArgumentParser(
        description="Export MossFormerGAN_SE_16K offline denoise model to ONNX "
        "with STFT/iSTFT baked in."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/data/user/lxp/llm/downloads/models/alibabasglab/MossFormerGAN_SE_16K",
        help="Path to the MossFormerGAN_SE_16K checkpoint directory.",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="debug/ClearerVoice-Studio/clearvoice",
        help="Path to a ClearerVoice-Studio clearvoice checkout providing "
        "clearvoice.models.mossformer_gan_se.generator.",
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default="public/models/mossformergan_se_16k.onnx",
        help="Output ONNX path.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=opset_version,
        help=f"ONNX opset version. Default: {opset_version}.",
    )
    parser.add_argument(
        "--skip-simplify",
        action="store_true",
        default=True,
        help="Skip onnxsim simplification (default: always skipped — see "
        "module docstring; onnxsim's shape inference cannot resolve this "
        "model's dynamic reshapes and raises a false-positive error).",
    )
    parser.add_argument(
        "--verify",
        type=int,
        default=1,
        help="Run PyTorch vs ONNX Runtime verification after export.",
    )
    parser.add_argument(
        "--quantize",
        type=int,
        default=1,
        help="Save dynamic int8 model next to ONNX using .onnx -> .int8.onnx.",
    )
    return parser.parse_args()


def load_mossformergan_se_16k(model_dir, source_dir):
    sys.path.insert(0, source_dir)
    import clearvoice.models.mossformer_gan_se.generator as generator_module
    from clearvoice.models.mossformer_gan_se.generator import SyncANet, SyncANetBlock

    class OnnxSyncANetBlock(SyncANetBlock):
        """
        SyncANetBlock.forward computes
        `T = math.ceil((old_T - emb_ks) / emb_hs) * emb_hs + emb_ks` from
        old_T = x.shape[2] (and the same for Q). math.ceil() is a plain
        Python builtin, not a torch op, so under trace-based ONNX export it
        collapses the traced dynamic size into a concrete Python int
        (the value at trace time) instead of an ONNX Shape/Gather sequence
        — every later `.view([B, T, ...])` in this method then bakes that
        one trace-time frame count into the graph, and any other input
        length fails at runtime with a Reshape shape-mismatch error.

        For SyncANet, emb_ks=2 and emb_hs=1 always (see SyncANet.__init__),
        so this formula is mathematically the identity: ceil((n-2)/1)*1+2 == n
        for any integer n. The padding it computes is therefore always
        exactly zero. This override skips the ceil arithmetic (and the
        resulting F.pad, per the same identity) and uses old_T/old_Q
        directly, which trace as genuinely dynamic sizes since they never
        pass through a non-tensor Python builtin. Behavior is unchanged;
        this only fixes ONNX export to generalize across input lengths.
        """

        def forward(self, x):
            B, C, old_T, old_Q = x.shape
            T, Q = old_T, old_Q

            input_ = x
            intra_rnn = self.intra_norm(input_)
            intra_rnn = self.Fconv(intra_rnn)
            intra_rnn = (
                intra_rnn.transpose(1, 2).contiguous().view(B * T, C * self.emb_ks, -1)
            )

            intra_rnn = intra_rnn.transpose(1, 2)
            intra_rnn_u = self.intra_to_u(intra_rnn)
            intra_rnn_v = self.intra_to_v(intra_rnn)
            intra_rnn_u = self.intra_rnn(intra_rnn_u)
            intra_rnn = intra_rnn_v * intra_rnn_u
            intra_rnn = intra_rnn.transpose(1, 2)
            intra_rnn = self.intra_linear(intra_rnn)
            intra_rnn = intra_rnn.transpose(1, 2)
            intra_rnn = intra_rnn.view([B, T, Q, C])
            intra_rnn = self.intra_mossformer(intra_rnn)
            intra_rnn = intra_rnn.transpose(1, 2)
            intra_rnn = intra_rnn.view([B, T, C, Q])
            intra_rnn = intra_rnn.transpose(1, 2).contiguous()
            intra_rnn = self.intra_se(intra_rnn)
            intra_rnn = intra_rnn + input_

            input_ = intra_rnn
            inter_rnn = self.inter_norm(input_)
            inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T)
            inter_rnn = F.unfold(
                inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
            )
            inter_rnn = inter_rnn.transpose(1, 2)
            inter_rnn_u = self.inter_to_u(inter_rnn)
            inter_rnn_v = self.inter_to_v(inter_rnn)
            inter_rnn_u = self.inter_rnn(inter_rnn_u)
            inter_rnn = inter_rnn_v * inter_rnn_u
            inter_rnn = inter_rnn.transpose(1, 2)
            inter_rnn = self.inter_linear(inter_rnn)
            inter_rnn = inter_rnn.transpose(1, 2)
            inter_rnn = inter_rnn.view([B, Q, T, C])
            inter_rnn = self.inter_mossformer(inter_rnn)
            inter_rnn = inter_rnn.transpose(1, 2)
            inter_rnn = inter_rnn.view([B, Q, C, T])
            inter_rnn = inter_rnn.permute(0, 2, 3, 1).contiguous()
            inter_rnn = self.inter_se(inter_rnn)
            inter_rnn = inter_rnn + input_

            inter_rnn = inter_rnn[..., :old_T, :old_Q]

            batch = inter_rnn
            all_Q, all_K, all_V = [], [], []
            for ii in range(self.n_head):
                all_Q.append(self[f"attn_conv_Q_{ii}"](batch))
                all_K.append(self[f"attn_conv_K_{ii}"](batch))
                all_V.append(self[f"attn_conv_V_{ii}"](batch))

            Qh = torch.cat(all_Q, dim=0)
            Kh = torch.cat(all_K, dim=0)
            Vh = torch.cat(all_V, dim=0)

            Qh = Qh.transpose(1, 2)
            Qh = Qh.flatten(start_dim=2)
            Kh = Kh.transpose(1, 2)
            Kh = Kh.flatten(start_dim=2)
            Vh = Vh.transpose(1, 2)
            old_shape = Vh.shape
            Vh = Vh.flatten(start_dim=2)
            emb_dim = Qh.shape[-1]

            attn_mat = torch.matmul(Qh, Kh.transpose(1, 2)) / (emb_dim**0.5)
            attn_mat = F.softmax(attn_mat, dim=2)
            Vh = torch.matmul(attn_mat, Vh)

            Vh = Vh.reshape(old_shape)
            Vh = Vh.transpose(1, 2)
            emb_dim = Vh.shape[1]

            batch = Vh.view([self.n_head, B, emb_dim, old_T, -1])
            batch = batch.transpose(0, 1)
            batch = batch.contiguous().view([B, self.n_head * emb_dim, old_T, -1])
            batch = self["attn_concat_proj"](batch)

            out = batch + inter_rnn
            return out

    generator_module.SyncANetBlock = OnnxSyncANetBlock

    class OnnxSyncANet(SyncANet):
        """
        SyncANet.forward computes `torch.angle(torch.complex(x[:, 0], x[:, 1]))`
        purely to get the phase of the input mag/phase pair; aten::complex has
        no ONNX symbolic in the exporter used here (opset 18, TorchScript
        exporter). Override forward with an identical copy that replaces
        that with torch.atan2(imag, real), which traces cleanly and is
        numerically identical (angle of complex(a, b) == atan2(b, a)).
        """

        def forward(self, x):
            out_list = []
            mag = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2).unsqueeze(1)
            noisy_phase = torch.atan2(x[:, 1, :, :], x[:, 0, :, :]).unsqueeze(1)
            x_in = torch.cat([mag, x], dim=1)

            x = self.dense_encoder(x_in)
            for ii in range(self.n_layers):
                x = self.blocks[ii](x)

            mask = self.mask_decoder(x)
            out_mag = mask * mag

            complex_out = self.complex_decoder(x)
            mag_real = out_mag * torch.cos(noisy_phase)
            mag_imag = out_mag * torch.sin(noisy_phase)
            final_real = mag_real + complex_out[:, 0, :, :].unsqueeze(1)
            final_imag = mag_imag + complex_out[:, 1, :, :].unsqueeze(1)
            out_list.append(final_real)
            out_list.append(final_imag)
            return out_list

    fft_len = 400
    model = OnnxSyncANet(num_channel=64, num_features=fft_len // 2 + 1)
    checkpoint_path = os.path.join(model_dir, "last_best_checkpoint.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"load_state_dict missing={missing} unexpected={unexpected}")
    return model.eval()


class ConvStft(nn.Module):
    """
    Conv-based reimplementation of torch.stft(center=True, onesided=True,
    return_complex=False) for a fixed (win_len, win_inc, fft_len, window).

    torch.onnx.export (opset 18, TorchScript exporter) cannot trace
    torch.stft/torch.istft/torch.complex/torch.angle for this graph
    (aten::complex has no ONNX symbolic; aten::istft requires a genuine
    complex tensor even eagerly). This reimplements the forward transform as
    reflect-pad + Conv1d against a fixed windowed DFT kernel, which traces
    cleanly and is verified below to match torch.stft to ~2e-5 max abs diff.
    """

    def __init__(self, win_len, win_inc, fft_len, window):
        super().__init__()
        self.win_inc = win_inc
        self.pad = fft_len // 2
        self.n_bins = fft_len // 2 + 1
        window_np = window.numpy().astype(np.float64)
        fourier_basis = np.fft.rfft(np.eye(fft_len))[:win_len]
        fwd_raw = np.concatenate([np.real(fourier_basis), np.imag(fourier_basis)], axis=1).T
        fwd_kernel = torch.from_numpy((fwd_raw * window_np).astype(np.float32)).unsqueeze(1)
        self.register_buffer("fwd_kernel", fwd_kernel)

    def forward(self, x):
        x = F.pad(x.unsqueeze(1), (self.pad, self.pad), mode="reflect")
        spec = F.conv1d(x, self.fwd_kernel, stride=self.win_inc)
        return spec[:, : self.n_bins], spec[:, self.n_bins :]


class ConviStft(nn.Module):
    """
    Conv-based reimplementation of torch.istft(center=True, onesided=True)
    for a fixed (win_len, win_inc, fft_len, window), companion to ConvStft.
    Uses pinv of the forward DFT matrix (correctly handling the onesided
    Hermitian doubling) for synthesis, then the standard overlap-add
    sum-of-squared-window normalization (matching torch.istft's NOLA
    algorithm, not a sqrt-window COLA pairing). Verified to match
    torch.istft to ~2e-5 max abs diff against the same spectrogram.
    """

    def __init__(self, win_len, win_inc, fft_len, window):
        super().__init__()
        self.win_inc = win_inc
        self.pad = fft_len // 2
        window_np = window.numpy().astype(np.float64)
        fourier_basis = np.fft.rfft(np.eye(fft_len))[:win_len]
        fwd_raw = np.concatenate([np.real(fourier_basis), np.imag(fourier_basis)], axis=1).T
        inv_raw = np.linalg.pinv(fwd_raw).T
        inv_kernel = torch.from_numpy((inv_raw * window_np).astype(np.float32)).unsqueeze(1)
        self.register_buffer("inv_kernel", inv_kernel)
        self.register_buffer("window_sq", window**2)
        self.register_buffer("enframe", torch.eye(win_len).unsqueeze(1))
        self.win_len = win_len

    def forward(self, real, imag, out_len):
        spec = torch.cat([real, imag], dim=1)
        wave = F.conv_transpose1d(spec, self.inv_kernel, stride=self.win_inc)
        win_sq = self.window_sq.reshape(1, self.win_len, 1).repeat(1, 1, spec.size(-1))
        coff = F.conv_transpose1d(win_sq, self.enframe, stride=self.win_inc)
        wave = (wave / (coff + 1e-8)).squeeze(1)
        return wave[:, self.pad : self.pad + out_len]


class MossformerganSe16kOnnxWrapper(nn.Module):
    win_len = 400
    win_inc = 100
    fft_len = 400

    def __init__(self, model):
        super().__init__()
        self.model = model
        window = torch.hamming_window(self.win_len, periodic=True)
        self.stft = ConvStft(self.win_len, self.win_inc, self.fft_len, window)
        self.istft = ConviStft(self.win_len, self.win_inc, self.fft_len, window)

    @staticmethod
    def power_compress(real, imag):
        mag = torch.sqrt(real**2 + imag**2) ** 0.3
        phase = torch.atan2(imag, real)
        return mag * torch.cos(phase), mag * torch.sin(phase)

    @staticmethod
    def power_uncompress(real, imag):
        mag = torch.sqrt(real**2 + imag**2) ** (1.0 / 0.3)
        phase = torch.atan2(imag, real)
        return mag * torch.cos(phase), mag * torch.sin(phase)

    def forward(self, speech):
        num_samples = speech.shape[-1]
        norm_factor = torch.sqrt(num_samples / torch.sum(speech**2, dim=-1))

        num_frames = torch.ceil(
            torch.tensor(num_samples, dtype=torch.float32) / self.win_inc
        )
        padded_len = (num_frames * self.win_inc).to(torch.int64)
        padding_len = padded_len - num_samples

        x = torch.cat([speech, speech[:, :padding_len]], dim=-1)
        x = x * norm_factor.unsqueeze(-1)

        real, imag = self.stft(x)
        real, imag = self.power_compress(real, imag)
        real_in = real.unsqueeze(1).permute(0, 1, 3, 2)
        imag_in = imag.unsqueeze(1).permute(0, 1, 3, 2)
        spec_in = torch.cat([real_in, imag_in], dim=1)

        out_list = self.model(spec_in)
        pred_real = out_list[0].permute(0, 1, 3, 2).squeeze(1)
        pred_imag = out_list[1].permute(0, 1, 3, 2).squeeze(1)
        unc_real, unc_imag = self.power_uncompress(pred_real, pred_imag)

        wav = self.istft(unc_real, unc_imag, padded_len)
        wav = wav / norm_factor.unsqueeze(-1)
        return wav[:, :num_samples]


def add_metadata_to_onnx(onnx_path, metadata_dict):
    model = onnx.load(onnx_path)
    del model.metadata_props[:]
    for key, value in metadata_dict.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)
    onnx.save(model, onnx_path)
    print(f"Added metadata: {metadata_dict}")


def simplify_onnx(onnx_path):
    model = onnx.load(onnx_path)
    model, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model, onnx_path)
    print("Simplified with onnxsim")


def dedupe_shared_gemm_weights(input_path, output_path):
    """
    Defensive measure carried over from the frcrn_se_16k exporter: ONNX
    Runtime's dynamic quantizer unconditionally rewrites every Gemm into
    MatMul(+Add) and, for transB=1, transposes the weight initializer in
    place, which corrupts the shape if two Gemm nodes share that
    initializer. Give each Gemm node's weight input its own private copy to
    avoid this regardless of whether this particular model happens to share
    any weights.
    """
    model = onnx.load(input_path)
    usage_count = {}
    for node in model.graph.node:
        if node.op_type == "Gemm" and len(node.input) > 1:
            usage_count[node.input[1]] = usage_count.get(node.input[1], 0) + 1

    initializers_by_name = {init.name: init for init in model.graph.initializer}
    seen_once = set()
    duplicated = 0
    for node in model.graph.node:
        if node.op_type != "Gemm" or len(node.input) <= 1:
            continue
        weight_name = node.input[1]
        if usage_count.get(weight_name, 0) <= 1:
            continue
        if weight_name not in seen_once:
            seen_once.add(weight_name)
            continue
        original = initializers_by_name[weight_name]
        new_init = onnx.TensorProto()
        new_init.CopyFrom(original)
        new_init.name = f"{weight_name}__dup{duplicated}"
        duplicated += 1
        model.graph.initializer.append(new_init)
        node.input[1] = new_init.name

    print(f"Duplicated {duplicated} shared Gemm weight initializers")
    onnx.save(model, output_path)


def quantize_onnx_model(input_path, output_path):
    from onnxruntime.quantization.shape_inference import quant_pre_process

    deduped_path = input_path.replace(".onnx", ".deduped.onnx")
    dedupe_shared_gemm_weights(input_path, deduped_path)

    preprocessed_path = input_path.replace(".onnx", ".preproc.onnx")
    quant_pre_process(deduped_path, preprocessed_path, skip_symbolic_shape=True)
    os.remove(deduped_path)

    model = onnx.load(preprocessed_path)
    nodes_to_exclude = []
    preprocess_keywords = ("window", "stft", "istft")

    preprocess_inits = [
        init.name
        for init in model.graph.initializer
        if any(keyword in init.name.lower() for keyword in preprocess_keywords)
    ]

    for node in model.graph.node:
        node_name = node.name.lower()
        if any(inp in preprocess_inits for inp in node.input):
            nodes_to_exclude.append(node.name)
            continue
        if any(keyword in node_name for keyword in preprocess_keywords):
            nodes_to_exclude.append(node.name)
            continue
        if node.op_type == "Conv":
            group = next((a.i for a in node.attribute if a.name == "group"), 1)
            if group != 1:
                # ONNX Runtime's CPU ConvInteger kernel does not support
                # grouped/depthwise convolution (UniDeepFsmn.conv1, and
                # DilatedDenseNet/SyncANetBlock's dilated depthwise convs,
                # all use groups != 1), so quantizing these produces a model
                # that fails to load at inference time.
                nodes_to_exclude.append(node.name)

    nodes_to_exclude = sorted(set(nodes_to_exclude))
    print(f"Excluding {len(nodes_to_exclude)} nodes from int8 quantization")

    # Restrict to Conv only: Gemm nodes are internally decomposed into
    # MatMul+Add by the quantizer before exclusion is applied, so a Gemm
    # node's original name never matches the synthesized MatMul node it
    # produces and nodes_to_exclude silently fails to protect it (see
    # frcrn_se_16k's exporter for the ShapeInferenceError this caused
    # there). Conv-only quantization sidesteps that class of bug entirely.
    quantize_dynamic(
        model_input=preprocessed_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,
        nodes_to_exclude=nodes_to_exclude,
        op_types_to_quantize=["Conv"],
        per_channel=False,
        reduce_range=False,
        extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
    )
    os.remove(preprocessed_path)
    print(f"Quantized int8 model saved to: {output_path}")


def verify_onnx(onnx_path, wrapper, num_samples_list):
    import numpy as np
    import onnxruntime as ort

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified OK")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    for num_samples in num_samples_list:
        speech = torch.randn(1, num_samples, dtype=torch.float32).clamp(-1.0, 1.0)
        with torch.no_grad():
            torch_out = wrapper(speech)

        ort_out = session.run(None, {"speech": speech.cpu().numpy()})[0]

        length = min(torch_out.shape[1], ort_out.shape[1])
        diff = float(
            np.max(np.abs(ort_out[:, :length] - torch_out.cpu().numpy()[:, :length]))
        )
        print(
            f"num_samples={num_samples}: torch_shape={tuple(torch_out.shape)} "
            f"onnx_shape={tuple(ort_out.shape)} max_abs_diff={diff:.8f}"
        )
        assert diff < 1e-3, f"enhanced diff too large at num_samples={num_samples}: {diff}"


def export_onnx(model_dir, source_dir, output_path, opset, skip_simplify, verify, quantize):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Missing model dir: {model_dir}")

    print(f"Loading MossFormerGAN_SE_16K from: {model_dir}")
    model = load_mossformergan_se_16k(model_dir, source_dir)
    wrapper = MossformerganSe16kOnnxWrapper(model).eval()

    dummy_speech = torch.randn(1, 16000, dtype=torch.float32).clamp(-1.0, 1.0)
    dummy_inputs = (dummy_speech,)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with patch_bool_eye():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            output_path,
            input_names=["speech"],
            output_names=["enhanced"],
            dynamic_axes={
                "speech": {1: "num_samples"},
                "enhanced": {1: "num_samples"},
            },
            opset_version=opset,
            verbose=False,
            dynamo=False,
        )
        print(f"Exported model to: {output_path}")

        if not skip_simplify:
            print(
                "Skipping onnxsim: its shape inference cannot resolve this "
                "model's dynamic reshapes (verified working without it; "
                "onnxsim raises a false-positive ShapeInferenceError at "
                "load time), unlike frcrn_se_16k.onnx's exporter."
            )

        metadata = {
            "model_type": "mossformergan_se_16k_denoise",
            "sample_rate": 16000,
            "input_scale": "normalized_float",
            "streaming": 0,
        }
        add_metadata_to_onnx(output_path, metadata)

        size = os.path.getsize(output_path)
        print(f"File size: {size:,} bytes ({size / 1024 / 1024:.2f} MB)")

        if verify:
            verify_onnx(output_path, wrapper, [16000, 24001])

    if quantize:
        quantized_path = output_path.replace(".onnx", ".int8.onnx")
        if quantized_path == output_path:
            raise ValueError(f"ONNX path must end with .onnx for int8 output: {output_path}")
        quantize_onnx_model(output_path, quantized_path)
        add_metadata_to_onnx(quantized_path, metadata)
        quantized_size = os.path.getsize(quantized_path)
        print(
            f"Int8 file size: {quantized_size:,} bytes "
            f"({quantized_size / 1024 / 1024:.2f} MB)"
        )


def main():
    args = get_args()
    export_onnx(
        model_dir=args.model_dir,
        source_dir=args.source_dir,
        output_path=args.onnx_path,
        opset=args.opset,
        skip_simplify=args.skip_simplify,
        verify=bool(args.verify),
        quantize=bool(args.quantize),
    )


if __name__ == "__main__":
    sys.exit(main())
