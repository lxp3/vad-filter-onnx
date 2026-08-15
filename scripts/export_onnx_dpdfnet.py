#!/usr/bin/env python3

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic


OPSET_VERSION = 18

# Each profile is a different upstream DPDFNet checkpoint family, differing
# only in sample rate / STFT frame size; the constructor args, streaming
# per-frame interface, and state layout are otherwise identical.
PROFILES = {
    "16khz": {
        "module": "dpdfnet",
        "cls": "DPDFNet",
        "sample_rate": 16000,
        "suffix": "",
    },
    "8khz": {
        "module": "dpdfnet_8khz",
        "cls": "DPDFNet8KHz",
        "sample_rate": 8000,
        "suffix": "_8khz",
    },
    "48khz_hr": {
        "module": "dpdfnet_48khz_hr",
        "cls": "DPDFNet48HR",
        "sample_rate": 48000,
        "suffix": "_48khz_hr",
    },
}


def vorbis_window(window_len: int) -> torch.Tensor:
    half = window_len / 2
    indices = torch.arange(window_len, dtype=torch.float32)
    s = torch.sin(0.5 * math.pi * (indices + 0.5) / half)
    return torch.sin(0.5 * math.pi * s * s)


class StreamingWaveformDPDFNet(nn.Module):
    def __init__(self, model: nn.Module, sample_rate: int):
        super().__init__()
        self.model = model
        self.sample_rate = sample_rate
        self.n_fft = int(model.stft.win_len)
        self.hop_size = int(model.stft.hop)

        samples = torch.arange(self.n_fft, dtype=torch.float32)
        frequencies = torch.arange(self.n_fft // 2 + 1, dtype=torch.float32)
        angles = 2.0 * math.pi * frequencies[:, None] * samples[None, :] / self.n_fft
        window = vorbis_window(self.n_fft)

        inverse_scale = torch.full((self.n_fft // 2 + 1,), 2.0 / self.n_fft)
        inverse_scale[0] = 1.0 / self.n_fft
        inverse_scale[-1] = 1.0 / self.n_fft

        wnorm = (
            float(model.wnorm)
            if hasattr(model, "wnorm")
            else 1.0 / (self.n_fft**2 / (2 * self.hop_size))
        )

        self.register_buffer("analysis_real", torch.cos(angles))
        self.register_buffer("analysis_imag", -torch.sin(angles))
        self.register_buffer("synthesis_real", torch.cos(angles) * inverse_scale[:, None])
        self.register_buffer("synthesis_imag", -torch.sin(angles) * inverse_scale[:, None])
        self.register_buffer("window", window)
        self.register_buffer("wnorm", torch.tensor(wnorm, dtype=torch.float32))
        self.register_buffer("inv_wnorm", torch.tensor(1.0 / wnorm, dtype=torch.float32))

    def forward(
        self,
        speech: torch.Tensor,
        analysis_cache: torch.Tensor,
        synthesis_cache: torch.Tensor,
        state_in: torch.Tensor,
    ):
        hop_size = self.hop_size
        frame = torch.cat([analysis_cache, speech], dim=1)
        windowed = frame * self.window
        real = torch.matmul(windowed, self.analysis_real.transpose(0, 1))
        imag = torch.matmul(windowed, self.analysis_imag.transpose(0, 1))
        spec = torch.stack([real, imag], dim=-1).unsqueeze(1) * self.wnorm

        spec_e, state_out = self.model(spec, state_in)
        spec_e = spec_e.squeeze(1) * self.inv_wnorm

        enhanced_frame = (
            torch.matmul(spec_e[..., 0], self.synthesis_real)
            + torch.matmul(spec_e[..., 1], self.synthesis_imag)
        ) * self.window

        enhanced = enhanced_frame[:, :hop_size] + synthesis_cache
        synthesis_cache = enhanced_frame[:, hop_size:]
        analysis_cache = speech
        return enhanced, analysis_cache, synthesis_cache, state_out


def parse_args():
    parser = argparse.ArgumentParser(description="Export streaming DPDFNet with waveform I/O")
    parser.add_argument("--source-dir", default="debug/DPDFNet")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="16khz")
    parser.add_argument("--dprnn-num-blocks", type=int, default=2)
    parser.add_argument(
        "--output",
        default=None,
        help="Defaults to public/models/dpdfnet<dprnn-num-blocks><profile-suffix>.onnx",
    )
    parser.add_argument("--verify-frames", type=int, default=20)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument(
        "--quantize",
        type=int,
        default=1,
        help="Save dynamic int8 model next to ONNX using .onnx -> .int8.onnx.",
    )
    args = parser.parse_args()
    if args.output is None:
        suffix = PROFILES[args.profile]["suffix"]
        args.output = f"public/models/dpdfnet{args.dprnn_num_blocks}{suffix}.onnx"
    return args


def load_model(source_dir: str, checkpoint: str, dprnn_num_blocks: int, profile: str):
    # onnxruntime.transformers stashes its own submodule under the bare name
    # "onnx_model" in sys.modules, which shadows this project's onnx_model
    # namespace package. Register the real package under that name explicitly
    # so the relative imports inside dpdfnet*.py/layers.py resolve correctly.
    import importlib
    import types

    source_root = Path(source_dir).resolve()
    package = types.ModuleType("onnx_model")
    package.__path__ = [str(source_root / "onnx_model")]
    sys.modules["onnx_model"] = package
    sys.path.insert(0, str(source_root))

    profile_info = PROFILES[profile]
    module = importlib.import_module(f"onnx_model.{profile_info['module']}")
    model_cls = getattr(module, profile_info["cls"])
    correct_state_dict = module.correct_state_dict
    from onnx_model.layers import convert_grouped_linear_to_einsum

    model = model_cls(
        conv_kernel_inp=(3, 3),
        conv_ch=64,
        enc_gru_dim=256,
        erb_dec_gru_dim=256,
        df_dec_gru_dim=256,
        enc_lin_groups=32,
        lin_groups=16,
        upsample_conv_type="subpixel",
        group_linear_type="loop",
        point_wise_type="cnn",
        separable_first_conv=True,
        dprnn_num_blocks=dprnn_num_blocks,
    )
    try:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state_dict = checkpoint_dict["state_dict"]
    model.load_state_dict(correct_state_dict(state_dict), strict=True)
    convert_grouped_linear_to_einsum(model)
    model.eval()

    return StreamingWaveformDPDFNet(model, sample_rate=profile_info["sample_rate"]).cpu().eval()


def initial_inputs(model: nn.Module):
    state_size = model.model.state_size()
    hop_size = model.hop_size
    return (
        torch.zeros(1, hop_size, dtype=torch.float32),
        torch.zeros(1, hop_size, dtype=torch.float32),
        torch.zeros(1, hop_size, dtype=torch.float32),
        torch.zeros(state_size, dtype=torch.float32),
    )


def add_metadata(
    output_path: str,
    state_size: int,
    variant: str,
    sample_rate: int,
    n_fft: int,
    hop_size: int,
):
    model = onnx.load(output_path)
    metadata = {
        "model_type": "dpdfnet_denoise",
        "variant": variant,
        "sample_rate": str(sample_rate),
        "frame_length": str(n_fft),
        "frame_shift": str(hop_size),
        "window_type": "vorbis",
        "state_size": str(state_size),
        "input_scale": "normalized_float",
        "streaming": "1",
    }
    del model.metadata_props[:]
    for key, value in metadata.items():
        item = model.metadata_props.add()
        item.key = key
        item.value = value
    onnx.checker.check_model(model)
    onnx.save(model, output_path)


def verify(model: nn.Module, output_path: str, num_frames: int):
    torch.manual_seed(20260815)
    hop_size = model.hop_size
    waveform = torch.rand(1, num_frames * hop_size, dtype=torch.float32) * 2.0 - 1.0
    torch_state = list(initial_inputs(model)[1:])
    ort_state = [value.numpy().copy() for value in initial_inputs(model)[1:]]
    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])

    waveform_diff = 0.0
    state_diff = 0.0
    for offset in range(0, waveform.shape[1], hop_size):
        speech = waveform[:, offset : offset + hop_size]
        with torch.no_grad():
            torch_outputs = model(speech, *torch_state)
        feeds = {
            "speech": speech.numpy(),
            "analysis_cache": ort_state[0],
            "synthesis_cache": ort_state[1],
            "state_in": ort_state[2],
        }
        ort_outputs = session.run(None, feeds)
        waveform_diff = max(
            waveform_diff,
            float(np.max(np.abs(torch_outputs[0].numpy() - ort_outputs[0]))),
        )
        for torch_cache, ort_cache in zip(torch_outputs[1:], ort_outputs[1:]):
            state_diff = max(
                state_diff,
                float(np.max(np.abs(torch_cache.numpy() - ort_cache))),
            )
        torch_state = [value.detach().clone() for value in torch_outputs[1:]]
        ort_state = [value.copy() for value in ort_outputs[1:]]

    print(f"PyTorch vs ONNX max abs diff: waveform={waveform_diff:.8g}, state={state_diff:.8g}")
    if waveform_diff >= 1e-4 or state_diff >= 1e-4:
        raise RuntimeError("PyTorch/ONNX maximum absolute difference exceeds 1e-4")


def quantize_onnx_model(input_path: str, output_path: str):
    model = onnx.load(input_path)
    nodes_to_exclude = []
    preprocess_inits = []
    preprocess_keywords = (
        "analysis_real",
        "analysis_imag",
        "synthesis_real",
        "synthesis_imag",
        "window",
        "wnorm",
        "inv_wnorm",
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
                # ONNX Runtime's CPU ConvInteger kernel does not support
                # grouped/depthwise convolution, so quantizing these
                # produces a model that fails to load at inference time.
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


def export(model: nn.Module, output_path: str, quantize: bool, dprnn_num_blocks: int, profile: str):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy_inputs = initial_inputs(model)
    torch.onnx.export(
        model,
        dummy_inputs,
        str(output),
        input_names=["speech", "analysis_cache", "synthesis_cache", "state_in"],
        output_names=["enhanced", "analysis_cache_out", "synthesis_cache_out", "state_out"],
        opset_version=OPSET_VERSION,
        dynamo=False,
    )
    state_size = model.model.state_size()
    variant = f"dpdfnet{dprnn_num_blocks}{PROFILES[profile]['suffix']}"
    add_metadata(str(output), state_size, variant, model.sample_rate, model.n_fft, model.hop_size)
    print(f"Exported DPDFNet waveform model to: {output}")

    if quantize:
        quantized_path = str(output).replace(".onnx", ".int8.onnx")
        if quantized_path == str(output):
            raise ValueError(f"ONNX path must end with .onnx for int8 output: {output}")
        quantize_onnx_model(str(output), quantized_path)
        add_metadata(
            quantized_path, state_size, variant, model.sample_rate, model.n_fft, model.hop_size
        )
        size = os.path.getsize(quantized_path)
        print(f"Int8 file size: {size:,} bytes ({size / 1024 / 1024:.2f} MB)")


def main():
    args = parse_args()
    if args.verify_frames <= 0:
        raise ValueError("--verify-frames must be greater than zero")
    model = load_model(args.source_dir, args.checkpoint, args.dprnn_num_blocks, args.profile)
    export(
        model,
        args.output,
        quantize=bool(args.quantize),
        dprnn_num_blocks=args.dprnn_num_blocks,
        profile=args.profile,
    )
    if not args.no_verify:
        verify(model, args.output, args.verify_frames)


if __name__ == "__main__":
    main()
