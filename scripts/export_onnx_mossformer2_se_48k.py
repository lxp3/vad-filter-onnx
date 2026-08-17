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
from torch import einsum

opset_version = 18


"""
MossFormer2_SE_48K offline denoise export notes
===================================================

MossFormer2_SE_48K (alibabasglab/MossFormer2_SE_48K) is a fully offline,
non-causal speech enhancement model. Its generator (TestNet/
MossFormer_MaskNet) predicts a real-valued time-frequency mask from a
180-channel Kaldi-style log-mel-filterbank (60 mel bins + delta +
delta-delta) feature; the mask is then applied identically to the real and
imaginary channels of a separate 1920/384/1920 STFT of the raw waveform,
and the enhanced waveform is recovered via iSTFT. This exporter bakes the
whole pipeline (feature extraction, masking, STFT/iSTFT) into a single
waveform-in/waveform-out ONNX graph, matching this project's
frcrn_se_16k.onnx/mossformergan_se_16k.onnx convention.

Audio format
------------
- Sample rate: 48000 Hz only.
- Input tensor name: speech.
- Input tensor shape: [1, num_samples] (dynamic time axis).
- Input dtype: float32, normalized float in [-1, 1] (scaled to PCM range
  internally, matching upstream's `inputs * MAX_WAV_VALUE` step).
- Output tensor name: enhanced.
- Output tensor shape: [1, num_samples] float32.

Kaldi fbank frontend (no torch.fft)
------------------------------------
- Upstream computes features via torchaudio.compliance.kaldi.fbank(...),
  which internally calls torch.fft.rfft — unsupported by this project's
  ONNX exporter (opset 18, TorchScript exporter): "Exporting the operator
  'aten::fft_rfft' is not supported". `KaldiFbank` below reimplements the
  exact same computation (framing, DC removal, preemphasis=0.97, Hamming
  window, power spectrum, mel filterbank, log) using a fixed matmul-based
  real DFT instead of torch.fft, reusing torchaudio's own internal helper
  functions (`_feature_window_function`, `get_mel_banks`) to build the
  window/mel-matrix constants so they are guaranteed identical to
  upstream's. dither=0 (upstream's own recommended setting when disabling
  dither is `energy_floor>0`, but this model doesn't use the energy
  channel at all, so dither is simply disabled for determinism, matching
  how this project's other exporters avoid randomness in the graph).
- Delta/delta-delta features reuse `torchaudio.functional.compute_deltas`
  directly (a fixed grouped Conv1d + replicate-pad — no FFT, traces fine).

STFT/iSTFT (no torch.stft/torch.istft)
-----------------------------------------
- The masking STFT/iSTFT (win_len=win_inc=... : win_len=1920, win_inc=384,
  fft_len=1920, Hamming, center=False/no padding — Kaldi "snip_edges"
  framing, same frame alignment as the fbank frontend) is reimplemented via
  Conv1d/ConvTranspose1d against a fixed windowed-DFT kernel (pinv-based
  synthesis kernel, same technique as frcrn_se_16k.onnx's ConvSTFT/
  ConviSTFT and mossformergan_se_16k.onnx's ConvStft/ConviStft), since
  torch.istft requires a genuine complex tensor which has no ONNX
  symbolic in this exporter.

Dynamic-length export fix
--------------------------
- `FLASH_ShareA_FFConvM.cal_attention` (mossformer2_block.py) computes
  `padding = padding_to_multiple_of(n, g)` then only pads
  `if padding > 0`. n = x.shape[-2] is a traced dynamic size, but
  `padding_to_multiple_of`'s `if remainder == 0: return 0` branches on a
  concrete Python bool derived from that size, and the subsequent
  `if padding > 0:` branches again — both bake a fixed decision (and a
  fixed pad amount) from whatever n was at trace time, breaking at any
  other input length. `load_mossformer2_se_48k` installs an
  `OnnxFlashShareAFFConvM` override (monkeypatched into the
  mossformer2_block module before the model is constructed) that always
  executes the branch-free equivalent `padding = (-n) % g` (0 when n is
  already a multiple of g, so unconditionally padding by this amount is a
  no-op in that case) — this never branches on a shape-derived value, so
  it traces as a genuinely dynamic Mod/Pad sequence.

Streaming
---------
- streaming=0 in ONNX metadata. This model must never be called
  incrementally; the C++ backend (Mossformer2Se48kDenoiseModel) buffers
  all input and only invokes the ONNX session once, on input_finished=true,
  segmenting long buffers the same way upstream's decode_window/
  one_time_decode_length segmenting does (4 s window, 75% stride, matching
  mossformergan_se_16k.onnx's C++ backend pattern).
"""


@contextlib.contextmanager
def patch_dynamic_group_padding():
    """See "Dynamic-length export fix" in the module docstring."""
    import clearvoice.models.mossformer2_se.mossformer2_block as block_module

    real_cls = block_module.FLASH_ShareA_FFConvM

    class OnnxFlashShareAFFConvM(real_cls):
        def cal_attention(self, x, quad_q, lin_q, quad_k, lin_k, v, u, mask=None):
            b, n, device, g = x.shape[0], x.shape[-2], x.device, self.group_size

            if self.rotary_pos_emb is not None:
                quad_q, lin_q, quad_k, lin_k = map(
                    self.rotary_pos_emb.rotate_queries_or_keys, (quad_q, lin_q, quad_k, lin_k)
                )

            padding = (-n) % g
            quad_q, quad_k, lin_q, lin_k, v, u = map(
                lambda t: F.pad(t, (0, 0, 0, padding), value=0.0),
                (quad_q, quad_k, lin_q, lin_k, v, u),
            )

            from einops import rearrange

            quad_q, quad_k, lin_q, lin_k, v, u = map(
                lambda t: rearrange(t, "b (g n) d -> b g n d", n=self.group_size),
                (quad_q, quad_k, lin_q, lin_k, v, u),
            )

            sim = einsum("... i d, ... j d -> ... i j", quad_q, quad_k) / g
            attn = F.relu(sim) ** 2
            attn = self.dropout(attn)

            if self.causal:
                causal_mask = torch.ones((g, g), dtype=torch.bool, device=device).triu(1)
                attn = attn.masked_fill(causal_mask, 0.0)

            quad_out_v = einsum("... i j, ... j d -> ... i d", attn, v)
            quad_out_u = einsum("... i j, ... j d -> ... i d", attn, u)

            if self.causal:
                lin_kv = einsum("b g n d, b g n e -> b g d e", lin_k, v) / g
                lin_kv = lin_kv.cumsum(dim=1)
                lin_kv = F.pad(lin_kv, (0, 0, 0, 0, 1, -1), value=0.0)
                lin_out_v = einsum("b g d e, b g n d -> b g n e", lin_kv, lin_q)

                lin_ku = einsum("b g n d, b g n e -> b g d e", lin_k, u) / g
                lin_ku = lin_ku.cumsum(dim=1)
                lin_ku = F.pad(lin_ku, (0, 0, 0, 0, 1, -1), value=0.0)
                lin_out_u = einsum("b g d e, b g n d -> b g n e", lin_ku, lin_q)
            else:
                lin_kv = einsum("b g n d, b g n e -> b d e", lin_k, v) / n
                lin_out_v = einsum("b g n d, b d e -> b g n e", lin_q, lin_kv)

                lin_ku = einsum("b g n d, b g n e -> b d e", lin_k, u) / n
                lin_out_u = einsum("b g n d, b d e -> b g n e", lin_q, lin_ku)

            return map(
                lambda t: rearrange(t, "b g n d -> b (g n) d")[:, :n],
                (quad_out_v + lin_out_v, quad_out_u + lin_out_u),
            )

    block_module.FLASH_ShareA_FFConvM = OnnxFlashShareAFFConvM
    try:
        yield
    finally:
        block_module.FLASH_ShareA_FFConvM = real_cls


def get_args():
    parser = argparse.ArgumentParser(
        description="Export MossFormer2_SE_48K offline denoise model to ONNX "
        "with fbank/STFT/iSTFT baked in."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/data/user/lxp/llm/downloads/models/alibabasglab/MossFormer2_SE_48K",
        help="Path to the MossFormer2_SE_48K checkpoint directory.",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="debug/ClearerVoice-Studio/clearvoice",
        help="Path to a ClearerVoice-Studio clearvoice checkout providing "
        "clearvoice.models.mossformer2_se.",
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default="public/models/mossformer2_se_48k.onnx",
        help="Output ONNX path.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=opset_version,
        help=f"ONNX opset version. Default: {opset_version}.",
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


def load_mossformer2_se_48k(model_dir, source_dir):
    sys.path.insert(0, source_dir)
    from clearvoice.models.mossformer2_se.mossformer2_se_wrapper import TestNet

    model = TestNet()
    checkpoint_path = os.path.join(model_dir, "last_best_checkpoint.pt")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"load_state_dict missing={missing} unexpected={unexpected}")
    return model.eval()


class KaldiFbank(nn.Module):
    """
    Reimplements torchaudio.compliance.kaldi.fbank(dither=0,
    remove_dc_offset=True, preemphasis_coefficient=0.97,
    window_type='hamming', snip_edges=True, use_power=True,
    use_log_fbank=True, use_energy=False, round_to_power_of_two=True)
    without torch.fft (see module docstring). Reuses torchaudio's own
    window/mel-matrix construction helpers so those constants are
    guaranteed identical to upstream's.
    """

    def __init__(self, sample_rate, frame_length_ms, frame_shift_ms, num_mel_bins):
        super().__init__()
        from torchaudio.compliance.kaldi import _feature_window_function, get_mel_banks

        self.window_shift = int(sample_rate * frame_shift_ms / 1000.0)
        self.window_size = int(sample_rate * frame_length_ms / 1000.0)
        self.padded_window_size = 1
        while self.padded_window_size < self.window_size:
            self.padded_window_size *= 2
        self.preemphasis_coefficient = 0.97

        window = _feature_window_function(
            "hamming", self.window_size, 0.0, torch.device("cpu"), torch.float32
        )
        self.register_buffer("window", window)

        mel_energies, _ = get_mel_banks(
            num_mel_bins, self.padded_window_size, float(sample_rate), 20.0, 0.0, 100.0, -500.0, 1.0
        )
        mel_energies = F.pad(mel_energies, (0, 1), mode="constant", value=0.0)
        self.register_buffer("mel_matrix", mel_energies.t().contiguous())

        n_bins = self.padded_window_size // 2 + 1
        fourier_basis = np.fft.rfft(np.eye(self.padded_window_size))[: self.padded_window_size]
        dft_real = torch.from_numpy(np.real(fourier_basis).astype(np.float32))
        dft_imag = torch.from_numpy(np.imag(fourier_basis).astype(np.float32))
        self.register_buffer("dft_real", dft_real)
        self.register_buffer("dft_imag", dft_imag)
        self.n_bins = n_bins

    def forward(self, waveform):
        # waveform: [num_samples] 1D float32, already PCM-scaled.
        num_samples = waveform.shape[0]
        num_frames = (num_samples - self.window_size) // self.window_shift + 1
        frame_idx = torch.arange(self.window_size, device=waveform.device)
        start_idx = torch.arange(num_frames, device=waveform.device) * self.window_shift
        gather_idx = (frame_idx.unsqueeze(0) + start_idx.unsqueeze(1)).reshape(-1)
        frames = waveform[gather_idx].reshape(num_frames, self.window_size)

        frames = frames - frames.mean(dim=1, keepdim=True)

        padded = F.pad(frames.unsqueeze(0), (1, 0), mode="replicate").squeeze(0)
        frames = frames - self.preemphasis_coefficient * padded[:, :-1]

        frames = frames * self.window
        frames = F.pad(frames, (0, self.padded_window_size - self.window_size))

        real = torch.matmul(frames, self.dft_real)
        imag = torch.matmul(frames, self.dft_imag)
        power = real**2 + imag**2

        mel = torch.matmul(power, self.mel_matrix)
        mel = torch.clamp(mel, min=1.1920928955078125e-07).log()
        return mel  # [num_frames, num_mel_bins]


class ConvStft(nn.Module):
    """
    Conv-based reimplementation of a non-centered (Kaldi snip_edges-style,
    no padding) torch.stft for a fixed (win_len, win_inc, fft_len, window).
    See mossformergan_se_16k's ConvStft/ConviStft for the centered variant
    and torch.onnx export rationale (aten::complex/aten::istft have no
    ONNX symbolic in this exporter).
    """

    def __init__(self, win_len, win_inc, fft_len, window):
        super().__init__()
        self.win_inc = win_inc
        self.n_bins = fft_len // 2 + 1
        window_np = window.numpy().astype(np.float64)
        fourier_basis = np.fft.rfft(np.eye(fft_len))[:win_len]
        fwd_raw = np.concatenate([np.real(fourier_basis), np.imag(fourier_basis)], axis=1).T
        fwd_kernel = torch.from_numpy((fwd_raw * window_np).astype(np.float32)).unsqueeze(1)
        self.register_buffer("fwd_kernel", fwd_kernel)

    def forward(self, x):
        spec = F.conv1d(x.unsqueeze(1), self.fwd_kernel, stride=self.win_inc)
        return spec[:, : self.n_bins], spec[:, self.n_bins :]


class ConviStft(nn.Module):
    """Companion to ConvStft; see mossformergan_se_16k's ConviStft."""

    def __init__(self, win_len, win_inc, fft_len, window):
        super().__init__()
        self.win_inc = win_inc
        self.win_len = win_len
        window_np = window.numpy().astype(np.float64)
        fourier_basis = np.fft.rfft(np.eye(fft_len))[:win_len]
        fwd_raw = np.concatenate([np.real(fourier_basis), np.imag(fourier_basis)], axis=1).T
        inv_raw = np.linalg.pinv(fwd_raw).T
        inv_kernel = torch.from_numpy((inv_raw * window_np).astype(np.float32)).unsqueeze(1)
        self.register_buffer("inv_kernel", inv_kernel)
        self.register_buffer("window_sq", window**2)
        self.register_buffer("enframe", torch.eye(win_len).unsqueeze(1))

    def forward(self, real, imag, out_len):
        spec = torch.cat([real, imag], dim=1)
        wave = F.conv_transpose1d(spec, self.inv_kernel, stride=self.win_inc)
        win_sq = self.window_sq.reshape(1, self.win_len, 1).repeat(1, 1, spec.size(-1))
        coff = F.conv_transpose1d(win_sq, self.enframe, stride=self.win_inc)
        wave = (wave / (coff + 1e-8)).squeeze(1)
        # No center-padding to trim here (unlike mossformergan_se_16k):
        # right-pad by a generous fixed amount then slice to out_len, since
        # the raw overlap-add length can be a few samples short of out_len
        # (snip_edges truncates any incomplete trailing frame) — mirrors
        # torch.istft's own `length=` right-zero-pad/truncate behavior.
        wave = F.pad(wave, (0, self.win_len))
        return wave[:, :out_len]


class Mossformer2Se48kOnnxWrapper(nn.Module):
    sample_rate = 48000
    frame_length_ms = 40
    frame_shift_ms = 8
    num_mel_bins = 60
    win_len = 1920
    win_inc = 384
    fft_len = 1920
    pcm_scale = 32768.0

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.fbank = KaldiFbank(
            self.sample_rate, self.frame_length_ms, self.frame_shift_ms, self.num_mel_bins
        )
        window = torch.hamming_window(self.win_len, periodic=False)
        self.stft = ConvStft(self.win_len, self.win_inc, self.fft_len, window)
        self.istft = ConviStft(self.win_len, self.win_inc, self.fft_len, window)
        self.compute_deltas = ComputeDeltas(self.num_mel_bins)

    def forward(self, speech):
        num_samples = speech.shape[-1]
        x = speech[0] * self.pcm_scale  # [num_samples], PCM scale like upstream.

        fbank_feat = self.fbank(x)  # [S, 60]
        fbank_tr = fbank_feat.transpose(0, 1).unsqueeze(0)  # [1, 60, S]
        delta = self.compute_deltas(fbank_tr)
        delta2 = self.compute_deltas(delta)
        feat = torch.cat([fbank_tr, delta, delta2], dim=1)  # [1, 180, S]
        feat = feat.transpose(1, 2)  # [1, S, 180], matches TestNet's expected layout.

        out_list = self.model(feat)
        mask = out_list[-1]  # [1, S, 961]
        mask = mask.transpose(1, 2)  # [1, 961, S]

        real, imag = self.stft(x.unsqueeze(0))  # [1, 961, S'] each
        frames = min(mask.shape[-1], real.shape[-1])
        mask = mask[:, :, :frames]
        real = real[:, :, :frames]
        imag = imag[:, :, :frames]

        masked_real = real * mask
        masked_imag = imag * mask
        enhanced = self.istft(masked_real, masked_imag, num_samples)
        enhanced = enhanced / self.pcm_scale
        return enhanced


class ComputeDeltas(nn.Module):
    """
    Reimplementation of torchaudio.functional.compute_deltas(win_length=5,
    mode="replicate") with a channel count fixed at construction time.
    Upstream builds its conv kernel via
    `torch.arange(-n, n+1).repeat(specgram.shape[1], 1, 1)`, i.e. a
    `.repeat()` whose count is a traced tensor size; the ONNX exporter
    then reports the resulting conv weight as having an "unknown shape",
    which ONNX's Conv requires to be static. Channel count is always 60
    here (num_mel_bins), so this precomputes the identical kernel as a
    plain constant buffer instead of via a traced repeat.
    """

    def __init__(self, channels, win_length=5):
        super().__init__()
        n = (win_length - 1) // 2
        denom = n * (n + 1) * (2 * n + 1) / 3
        kernel_1d = torch.arange(-n, n + 1, dtype=torch.float32) / denom
        kernel = kernel_1d.view(1, 1, -1).repeat(channels, 1, 1)
        self.register_buffer("kernel", kernel)
        self.n = n
        self.channels = channels

    def forward(self, x):
        x = F.pad(x, (self.n, self.n), mode="replicate")
        return F.conv1d(x, self.kernel, groups=self.channels)


def add_metadata_to_onnx(onnx_path, metadata_dict):
    model = onnx.load(onnx_path)
    del model.metadata_props[:]
    for key, value in metadata_dict.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)
    onnx.save(model, onnx_path)
    print(f"Added metadata: {metadata_dict}")


def quantize_onnx_model(input_path, output_path):
    from onnxruntime.quantization.shape_inference import quant_pre_process

    preprocessed_path = input_path.replace(".onnx", ".preproc.onnx")
    quant_pre_process(input_path, preprocessed_path, skip_symbolic_shape=True)

    model = onnx.load(preprocessed_path)
    nodes_to_exclude = []
    preprocess_keywords = ("window", "stft", "istft", "fbank", "mel", "dft")

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
                nodes_to_exclude.append(node.name)

    nodes_to_exclude = sorted(set(nodes_to_exclude))
    print(f"Excluding {len(nodes_to_exclude)} nodes from int8 quantization")

    # Conv-only, same rationale as frcrn_se_16k/mossformergan_se_16k: Gemm
    # nodes are internally decomposed into MatMul+Add by the quantizer
    # regardless of exclusion, and shared-weight Gemms can get corrupted by
    # that decomposition (see frcrn_se_16k's exporter for the bug this
    # caused there).
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

        length = min(torch_out.shape[-1], ort_out.shape[-1])
        diff = float(np.max(np.abs(ort_out[..., :length] - torch_out.cpu().numpy()[..., :length])))
        print(
            f"num_samples={num_samples}: torch_shape={tuple(torch_out.shape)} "
            f"onnx_shape={tuple(ort_out.shape)} max_abs_diff={diff:.8f}"
        )
        assert diff < 1e-3, f"enhanced diff too large at num_samples={num_samples}: {diff}"


def export_onnx(model_dir, source_dir, output_path, opset, verify, quantize):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Missing model dir: {model_dir}")

    sys.path.insert(0, source_dir)
    with patch_dynamic_group_padding():
        print(f"Loading MossFormer2_SE_48K from: {model_dir}")
        model = load_mossformer2_se_48k(model_dir, source_dir)
        wrapper = Mossformer2Se48kOnnxWrapper(model).eval()

        dummy_speech = torch.randn(1, 48000, dtype=torch.float32).clamp(-1.0, 1.0)
        dummy_inputs = (dummy_speech,)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
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

        metadata = {
            "model_type": "mossformer2_se_48k_denoise",
            "sample_rate": 48000,
            "input_scale": "normalized_float",
            "streaming": 0,
        }
        add_metadata_to_onnx(output_path, metadata)

        size = os.path.getsize(output_path)
        print(f"File size: {size:,} bytes ({size / 1024 / 1024:.2f} MB)")

        if verify:
            verify_onnx(output_path, wrapper, [48000, 72001])

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
        verify=bool(args.verify),
        quantize=bool(args.quantize),
    )


if __name__ == "__main__":
    sys.exit(main())
