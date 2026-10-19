#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PULSEVAD_ROOT = os.path.join(REPO_ROOT, "debug", "PulseVAD")
PULSEVAD_DATA = os.path.join(PULSEVAD_ROOT, "pulsevad", "data")

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 3200
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 64
N_FRAMES = 21
PREEMPHASIS_ALPHA = 0.97
EPS = 1e-5


"""
PulseVAD ONNX export
====================

Bakes the official NumPy frontend into the graph so C++ callers pass raw
waveform samples. Two CNN widths are exported from the local PulseVAD clone:

- 2.1k: BN-folded pruned student (`pulsevad_2.1k.pth`)
- 81k: unpruned teacher, BN-folded at export time (`pulsevad_teacher_81k.pth`)

Audio format
------------
- Sample rate: 16000 Hz only.
- Input tensor name: speech.
- Input tensor shape: [batch, 3200].
- Input dtype: float32 in [-1, 1].

Frontend (matches pulsevad.frontend_np.extract_log_mel_np)
----------------------------------------------------------
pre-emphasis 0.97 -> waveform z-norm -> reflect-padded STFT
(n_fft=512, win=400 periodic Hann, hop=160, center=True) -> 64-bin HTK/Slaney
mel -> log(+1e-5) -> per-bin z-norm. Output features are [batch, 64, 21].

CNN
---
FoldedPulseVAD: depthwise-separable 1D conv stack + GAP + Linear(2).
No recurrent cache. One speech probability per 200 ms window.

Inputs
------
- speech [batch, 3200] float32.

Outputs
-------
- probs [batch, 1] float32, sigmoid(logit_speech - logit_nonspeech).

Post-processing stays outside the graph. C++ uses frame_length=3200 and
frame_shift=1600 (100 ms hop, matching official get_speech_timestamps).
"""


def get_args():
    parser = argparse.ArgumentParser(
        description="Export PulseVAD 2.1k / 81k to ONNX with mel frontend baked in."
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=PULSEVAD_ROOT,
        help="Path to the local PulseVAD clone.",
    )
    parser.add_argument(
        "--onnx-dir",
        type=str,
        default=os.path.join(REPO_ROOT, "public", "models"),
        help="Directory for exported ONNX files.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=("2.1k", "81k", "all"),
        help="Which PulseVAD width to export.",
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
        help="Verify ONNX vs PyTorch / official PulseVAD (1=yes, 0=no).",
    )
    parser.add_argument(
        "--quantize",
        type=int,
        default=1,
        help="Also export dynamically quantized int8 model (1=yes, 0=no).",
    )
    return parser.parse_args()


class PulseVadMelFrontend(nn.Module):
    """Raw waveform [B, 3200] -> normalized log-Mel [B, 64, 21]."""

    def __init__(self, mel_filters: torch.Tensor):
        super().__init__()
        self.register_buffer("mel_filters", mel_filters.float())  # [257, 64]

        hann = torch.hann_window(WIN_LENGTH, periodic=True, dtype=torch.float32)
        wpad = (N_FFT - WIN_LENGTH) // 2
        self.register_buffer("window", F.pad(hann, (wpad, N_FFT - WIN_LENGTH - wpad)))

        n = torch.arange(N_FFT, dtype=torch.float32).unsqueeze(1)
        n_stft = N_FFT // 2 + 1
        k = torch.arange(n_stft, dtype=torch.float32).unsqueeze(0)
        angles = 2 * math.pi * k * n / N_FFT
        self.register_buffer("dft_real", torch.cos(angles))
        self.register_buffer("dft_imag", -torch.sin(angles))

    def forward(self, speech: torch.Tensor) -> torch.Tensor:
        x = torch.cat(
            [speech[:, :1], speech[:, 1:] - PREEMPHASIS_ALPHA * speech[:, :-1]], dim=1
        )
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / (std + EPS)

        pad = N_FFT // 2
        x = F.pad(x, (pad, pad), mode="reflect")
        frame_idx = torch.arange(N_FFT, device=x.device).unsqueeze(0)
        start_idx = (torch.arange(N_FRAMES, device=x.device) * HOP_LENGTH).unsqueeze(1)
        gather_idx = (frame_idx + start_idx).reshape(-1)
        frames = x[:, gather_idx].reshape(x.shape[0], N_FRAMES, N_FFT)
        windowed = frames * self.window
        spec_real = torch.matmul(windowed, self.dft_real)
        spec_imag = torch.matmul(windowed, self.dft_imag)
        power = spec_real.pow(2) + spec_imag.pow(2)
        mel = torch.matmul(power, self.mel_filters)
        log_mel = torch.log(mel + EPS).transpose(1, 2)

        mean = log_mel.mean(dim=-1, keepdim=True)
        std = log_mel.std(dim=-1, keepdim=True, unbiased=False)
        return (log_mel - mean) / (std + EPS)


class PulseVadWrapper(nn.Module):
    def __init__(self, frontend: PulseVadMelFrontend, cnn: nn.Module):
        super().__init__()
        self.frontend = frontend
        self.cnn = cnn

    def forward(self, speech: torch.Tensor) -> torch.Tensor:
        feat = self.frontend(speech)
        logits = self.cnn(feat)
        return torch.sigmoid(logits[:, 1:2] - logits[:, 0:1])


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
    model, check = simplify(model, dynamic_input_shape=True)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model, onnx_path)
    print("Simplified with onnxsim")


def quantize_onnx_model(input_path, output_path):
    model = onnx.load(input_path)
    nodes_to_exclude = []
    preprocess_inits = []
    preprocess_keywords = (
        "frontend",
        "fbank",
        "mel",
        "dft",
        "window",
    )

    for init in model.graph.initializer:
        if any(keyword in init.name.lower() for keyword in preprocess_keywords):
            preprocess_inits.append(init.name)

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

    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,
        nodes_to_exclude=nodes_to_exclude,
        per_channel=False,
        reduce_range=False,
    )
    print(f"Quantized int8 model saved to: {output_path}")


def _max_abs_diff(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def load_cnn(variant: str, source_dir: str):
    from pulsevad.model import PulseVAD
    from pulsevad.quantize import FoldedPulseVAD, fold_batchnorm

    data_dir = os.path.join(source_dir, "pulsevad", "data")
    if variant == "2.1k":
        ck_path = os.path.join(data_dir, "pulsevad_2.1k.pth")
        official_onnx = os.path.join(data_dir, "pulsevad_2.1k.onnx")
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        cnn = FoldedPulseVAD(ck["dims"]).eval()
        cnn.load_state_dict(ck["state_dict"])
        return cnn, official_onnx
    if variant == "81k":
        ck_path = os.path.join(data_dir, "pulsevad_teacher_81k.pth")
        official_onnx = os.path.join(data_dir, "pulsevad_teacher_81k.onnx")
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        teacher = PulseVAD().eval()
        teacher.load_state_dict(ck["state_dict"])
        cnn = fold_batchnorm(teacher).eval()
        return cnn, official_onnx
    raise ValueError(f"Unknown variant: {variant}")


def verify_frontend(frontend: PulseVadMelFrontend, speech_np: np.ndarray):
    from pulsevad.frontend_np import extract_log_mel_np

    ref = extract_log_mel_np(speech_np)
    with torch.no_grad():
        got = frontend(torch.from_numpy(speech_np)).cpu().numpy()
    diff = _max_abs_diff(ref, got)
    print(f"Frontend vs extract_log_mel_np max abs diff: {diff:.8f}")
    assert diff < 1e-4, f"frontend diff too large: {diff}"


def verify_cnn(cnn: nn.Module, official_onnx: str):
    import onnxruntime as ort

    x = torch.randn(4, N_MELS, N_FRAMES, dtype=torch.float32)
    with torch.no_grad():
        torch_logits = cnn(x).cpu().numpy()
    sess = ort.InferenceSession(official_onnx, providers=["CPUExecutionProvider"])
    ort_logits = sess.run(["logits"], {"log_mel": x.numpy()})[0]
    diff = _max_abs_diff(torch_logits, ort_logits)
    print(f"CNN vs official ONNX logits max abs diff: {diff:.8f}")
    assert diff < 1e-4, f"cnn logits diff too large: {diff}"


def verify_wrapper(onnx_path, wrapper: PulseVadWrapper, speech: torch.Tensor, official_onnx: str):
    import onnxruntime as ort
    from pulsevad.utils_vad import predict_window

    onnx.checker.check_model(onnx.load(onnx_path))
    print("ONNX model verified OK")

    with torch.no_grad():
        torch_probs = wrapper(speech).cpu().numpy()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_probs = session.run(None, {"speech": speech.cpu().numpy()})[0]
    diff = _max_abs_diff(torch_probs, ort_probs)
    print(f"PyTorch vs exported ONNX max abs diff: probs={diff:.8f}")
    assert diff < 1e-4, f"probs diff too large: {diff}"

    official = ort.InferenceSession(official_onnx, providers=["CPUExecutionProvider"])
    official_probs = []
    for row in speech.cpu().numpy():
        official_probs.append(predict_window(official, row))
    official_probs = np.asarray(official_probs, dtype=np.float32).reshape(-1, 1)
    official_diff = _max_abs_diff(ort_probs, official_probs)
    print(f"Exported ONNX vs official predict_window max abs diff: {official_diff:.8f}")
    assert official_diff < 1e-4, f"official predict_window diff too large: {official_diff}"
    return diff, official_diff


def export_variant(variant, source_dir, onnx_dir, opset, skip_simplify, verify, quantize):
    data_dir = os.path.join(source_dir, "pulsevad", "data")
    mel_path = os.path.join(data_dir, "mel_filterbank.npy")
    if not os.path.isfile(mel_path):
        raise FileNotFoundError(f"Missing mel filterbank: {mel_path}")

    cnn, official_onnx = load_cnn(variant, source_dir)
    mel_filters = torch.from_numpy(np.load(mel_path).astype(np.float32))
    frontend = PulseVadMelFrontend(mel_filters).eval()
    wrapper = PulseVadWrapper(frontend, cnn).cpu().eval()

    dummy_speech = torch.randn(1, WINDOW_SAMPLES, dtype=torch.float32).clamp(-1.0, 1.0)
    filename = "pulsevad.onnx" if variant == "2.1k" else "pulsevad_81k.onnx"
    output_path = os.path.join(onnx_dir, filename)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy_speech,
        output_path,
        input_names=["speech"],
        output_names=["probs"],
        dynamic_axes={"speech": {0: "batch"}, "probs": {0: "batch"}},
        opset_version=opset,
        verbose=False,
        dynamo=False,
    )
    print(f"Exported {variant} model to: {output_path}")

    if not skip_simplify:
        simplify_onnx(output_path)

    metadata = {
        "model_type": "pulsevad",
        "variant": variant,
        "sample_rate": SAMPLE_RATE,
        "window_samples": WINDOW_SAMPLES,
        "hop_samples": WINDOW_SAMPLES // 2,
        "input_scale": "normalized_float",
        "streaming": 0,
    }
    add_metadata_to_onnx(output_path, metadata)

    size = os.path.getsize(output_path)
    print(f"File size: {size:,} bytes ({size / 1024 / 1024:.2f} MB)")

    diffs = {"probs": None, "official": None, "int8": None}
    if verify:
        verify_speech = torch.randn(4, WINDOW_SAMPLES, dtype=torch.float32).clamp(-1.0, 1.0)
        verify_frontend(frontend, verify_speech.numpy())
        verify_cnn(cnn, official_onnx)
        diffs["probs"], diffs["official"] = verify_wrapper(
            output_path, wrapper, verify_speech, official_onnx
        )

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
        if verify:
            import onnxruntime as ort

            with torch.no_grad():
                torch_probs = wrapper(dummy_speech).cpu().numpy()
            sess = ort.InferenceSession(quantized_path, providers=["CPUExecutionProvider"])
            int8_probs = sess.run(None, {"speech": dummy_speech.cpu().numpy()})[0]
            diffs["int8"] = _max_abs_diff(torch_probs, int8_probs)
            print(f"PyTorch vs int8 ONNX max abs diff: probs={diffs['int8']:.8f}")

    return diffs


def main():
    args = get_args()
    source_dir = os.path.abspath(args.source_dir)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    if not os.path.isdir(os.path.join(source_dir, "pulsevad")):
        raise FileNotFoundError(f"PulseVAD source not found: {source_dir}")

    variants = ("2.1k", "81k") if args.variant == "all" else (args.variant,)
    for variant in variants:
        print(f"\n=== Exporting PulseVAD {variant} ===")
        export_variant(
            variant=variant,
            source_dir=source_dir,
            onnx_dir=args.onnx_dir,
            opset=args.opset,
            skip_simplify=args.skip_simplify,
            verify=bool(args.verify),
            quantize=bool(args.quantize),
        )


if __name__ == "__main__":
    sys.exit(main())
