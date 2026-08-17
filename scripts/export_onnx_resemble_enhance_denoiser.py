#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
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

"""
resemble-enhance Denoiser offline export notes
================================================

resemble-enhance (resemble-ai/resemble-enhance) ships two independently
runnable sub-models: a `Denoiser` (STFT-domain masking U-Net) and a much
heavier `Enhancer` (denoiser + latent conditional-flow-matching generative
restoration + UnivNet vocoder, 16-64 ODE-solver forward passes per output).
Only `Denoiser` is exported here: it is a plain offline, non-causal,
single-shot waveform-in/waveform-out model (STFT -> 2D-conv U-Net mask ->
iSTFT), architecturally the same shape as frcrn_se_16k.onnx and
mossformergan_se_16k.onnx already in this project. `Enhancer`'s CFM+vocoder
stages are a separate, much more expensive generative pipeline and are not
exported by this script.

Audio format
------------
- Sample rate: 44100 Hz only (hardcoded in resemble_enhance.hparams.HParams).
- Input tensor name: speech.
- Input tensor shape: [1, num_samples] (dynamic).
- Input dtype: float32, arbitrary-scale waveform (normalized internally).
- Output tensor name: enhanced.
- Output tensor shape: [1, num_samples] float32.

Model internals
----------------
- STFT: hop_size=420 (~9.5 ms), win_length=n_fft=1680 (~38 ms), centered
  (reflect-padded), periodic Hann window, magnitude/phase (cos, sin)
  features -- not raw real/imag.
- `Denoiser._stft`/`_istft` (denoiser/denoiser.py) use torch.stft/istft with
  return_complex=True and torch.complex(...)/.angle(); aten::complex has no
  ONNX symbolic in the TorchScript exporter used here (opset 18), so this
  wrapper reimplements STFT/iSTFT as fixed windowed-DFT Conv1d/
  ConvTranspose1d kernels (the same ConvStft/ConviStft pattern used by
  frcrn_se_16k.onnx and mossformergan_se_16k.onnx), and replaces
  torch.angle(torch.complex(re, im)) with torch.atan2(im, re) (identical).
- Denoiser._stft drops the last STFT frame (`s[..., :-1]`) before feeding
  the U-Net, then Denoiser._istft pads one frame back via
  `F.pad(s, (0, 1), "replicate")` before iSTFT. This wrapper replicates that
  exactly with a real-valued last-column replicate pad.
- The U-Net (denoiser/unet.py) is a plain 4-level 2D-conv U-Net
  (Conv2d/GroupNorm/GELU/Upsample only) with no attention, no eye, no
  linalg, and no data-dependent Python branching on traced tensor sizes --
  it exports cleanly with no monkeypatches required.
- No recurrent cache: the whole utterance is processed in one shot.

Streaming
---------
- streaming=0 in ONNX metadata. resemble-enhance's Denoiser (and the full
  Enhancer) has no causal/frame-wise state anywhere; it does NOT support
  streaming or low-latency real-time operation. Upstream's own "chunking"
  (resemble_enhance/inference.py) is fixed-size 30s/1s-overlap batch
  splitting with cross-correlation realignment and crossfade purely to
  bound memory use on long files -- not a real-time streaming mode. The
  C++ backend (ResembleEnhanceDenoiserDenoiseModel) buffers all input and
  only invokes the ONNX session once, on input_finished=true, mirroring
  frcrn_se_16k.onnx and mossformergan_se_16k.onnx's non-streaming backends.
"""


def load_denoiser(model_dir):
    # Import from denoiser.denoiser/denoiser.hparams directly (not
    # denoiser.train), since train.py unconditionally imports deepspeed,
    # which is not needed for inference/export and may not be installed.
    from resemble_enhance.denoiser.denoiser import Denoiser
    from resemble_enhance.denoiser.hparams import HParams

    hp = HParams()
    model = Denoiser(hp)
    checkpoint_path = os.path.join(model_dir, "enhancer_stage2", "ds", "G", "default",
                                    "mp_rank_00_model_states.pt")
    state_dict = torch.load(checkpoint_path, map_location="cpu")["module"]
    denoiser_state = {
        key[len("denoiser."):]: value
        for key, value in state_dict.items()
        if key.startswith("denoiser.")
    }
    missing, unexpected = model.load_state_dict(denoiser_state, strict=False)
    if missing or unexpected:
        print(f"load_state_dict missing={missing} unexpected={unexpected}")
    return model.eval()


class ConvStft(nn.Module):
    """
    Conv-based reimplementation of torch.stft(center=True, onesided=True,
    return_complex=False) for a fixed (win_len, win_inc, fft_len, window).
    See mossformergan_se_16k's exporter for why torch.stft/complex ops
    cannot be traced to ONNX here. Verified below to match torch.stft to
    ~1e-5 max abs diff.
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
    Conv-based reimplementation of torch.istft(center=True, onesided=True),
    companion to ConvStft. See mossformergan_se_16k's exporter for details.
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


class ResembleEnhanceDenoiserOnnxWrapper(nn.Module):
    hop_size = 420
    win_len = hop_size * 4
    fft_len = hop_size * 4
    eps = 1e-7

    def __init__(self, model):
        super().__init__()
        self.net = model.net
        window = torch.hann_window(self.win_len, periodic=True)
        self.stft = ConvStft(self.win_len, self.hop_size, self.fft_len, window)
        self.istft = ConviStft(self.win_len, self.hop_size, self.fft_len, window)

    @staticmethod
    def _magphase(real, imag, eps):
        mag = torch.sqrt(real**2 + imag**2 + eps)
        return mag, real / mag, imag / mag

    def forward(self, speech):
        num_samples = speech.shape[-1]
        abs_max = speech.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-7)
        x = speech / abs_max

        real, imag = self.stft(x)
        # Denoiser._stft drops the last (centered) STFT frame before
        # feeding the U-Net; Denoiser._istft pads it back afterward via a
        # replicate pad on the last column. Mirrored exactly here.
        real = real[..., :-1]
        imag = imag[..., :-1]
        mag, cos, sin = self._magphase(real, imag, self.eps)

        feat = torch.stack([mag, cos, sin], dim=1)
        mag_mask, res_real, res_imag = self.net(feat).unbind(1)
        mag_mask = mag_mask.sigmoid()
        res_real = res_real.tanh()
        res_imag = res_imag.tanh()
        _, cos_res, sin_res = self._magphase(res_real, res_imag, self.eps)

        sep_mag = F.relu(mag * mag_mask)
        sep_real = sep_mag * (cos * cos_res - sin * sin_res)
        sep_imag = sep_mag * (sin * cos_res + cos * sin_res)

        sep_real = F.pad(sep_real, (0, 1), mode="replicate")
        sep_imag = F.pad(sep_imag, (0, 1), mode="replicate")

        out = self.istft(sep_real, sep_imag, num_samples)
        out = out * abs_max
        return out


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


def quantize_onnx_model(input_path, output_path):
    from onnxruntime.quantization.shape_inference import quant_pre_process

    preprocessed_path = input_path.replace(".onnx", ".preproc.onnx")
    quant_pre_process(input_path, preprocessed_path, skip_symbolic_shape=True)

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

    nodes_to_exclude = sorted(set(nodes_to_exclude))
    print(f"Excluding {len(nodes_to_exclude)} nodes from int8 quantization")

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

        length = min(torch_out.shape[1], ort_out.shape[1])
        diff = float(
            np.max(np.abs(ort_out[:, :length] - torch_out.cpu().numpy()[:, :length]))
        )
        print(
            f"num_samples={num_samples}: torch_shape={tuple(torch_out.shape)} "
            f"onnx_shape={tuple(ort_out.shape)} max_abs_diff={diff:.8f}"
        )
        assert diff < 1e-3, f"enhanced diff too large at num_samples={num_samples}: {diff}"


def export_onnx(model_dir, output_path, opset, skip_simplify, verify, quantize):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Missing model dir: {model_dir}")

    print(f"Loading resemble-enhance Denoiser from: {model_dir}")
    model = load_denoiser(model_dir)
    wrapper = ResembleEnhanceDenoiserOnnxWrapper(model).eval()

    dummy_speech = torch.randn(1, 44100, dtype=torch.float32).clamp(-1.0, 1.0)
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

    if not skip_simplify:
        simplify_onnx(output_path)

    metadata = {
        "model_type": "resemble_enhance_denoiser_denoise",
        "sample_rate": 44100,
        "input_scale": "arbitrary_float",
        "streaming": 0,
    }
    add_metadata_to_onnx(output_path, metadata)

    size = os.path.getsize(output_path)
    print(f"File size: {size:,} bytes ({size / 1024 / 1024:.2f} MB)")

    if verify:
        verify_onnx(output_path, wrapper, [44100, 66151])

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


def get_args():
    parser = argparse.ArgumentParser(
        description="Export resemble-enhance's Denoiser offline denoise model to ONNX "
        "with STFT/iSTFT baked in."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/data/user/lxp/llm/downloads/models/ResembleAI/resemble-enhance",
        help="Path to the resemble-enhance checkpoint directory (HuggingFace clone).",
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default="public/models/resemble_enhance_denoiser.onnx",
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
        help="Skip onnxsim simplification.",
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


def main():
    args = get_args()
    export_onnx(
        model_dir=args.model_dir,
        output_path=args.onnx_path,
        opset=args.opset,
        skip_simplify=args.skip_simplify,
        verify=bool(args.verify),
        quantize=bool(args.quantize),
    )


if __name__ == "__main__":
    sys.exit(main())
