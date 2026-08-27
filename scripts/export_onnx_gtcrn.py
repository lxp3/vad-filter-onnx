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
from onnxconverter_common import float16
from onnxruntime.quantization import QuantType, quantize_dynamic


NFFT = 512
HOP_SIZE = 256
SAMPLE_RATE = 16000
OPSET_VERSION = 18


class StreamingWaveformGTCRN(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        samples = torch.arange(NFFT, dtype=torch.float32)
        frequencies = torch.arange(NFFT // 2 + 1, dtype=torch.float32)
        angles = 2.0 * math.pi * frequencies[:, None] * samples[None, :] / NFFT
        window = torch.hann_window(NFFT, periodic=True).sqrt()

        inverse_scale = torch.full((NFFT // 2 + 1,), 2.0 / NFFT)
        inverse_scale[0] = 1.0 / NFFT
        inverse_scale[-1] = 1.0 / NFFT

        self.register_buffer("analysis_real", torch.cos(angles))
        self.register_buffer("analysis_imag", -torch.sin(angles))
        self.register_buffer("synthesis_real", torch.cos(angles) * inverse_scale[:, None])
        self.register_buffer("synthesis_imag", -torch.sin(angles) * inverse_scale[:, None])
        self.register_buffer("window", window)

    def forward(
        self,
        speech: torch.Tensor,
        conv_cache: torch.Tensor,
        tra_cache: torch.Tensor,
        inter_cache: torch.Tensor,
        analysis_cache: torch.Tensor,
        synthesis_cache: torch.Tensor,
    ):
        frame = torch.cat([analysis_cache, speech], dim=1)
        windowed = frame * self.window
        real = torch.matmul(windowed, self.analysis_real.transpose(0, 1))
        imag = torch.matmul(windowed, self.analysis_imag.transpose(0, 1))
        spectrum = torch.stack([real, imag], dim=-1).unsqueeze(2)

        enhanced_spectrum, conv_cache, tra_cache, inter_cache = self.model(
            spectrum, conv_cache, tra_cache, inter_cache
        )
        enhanced_spectrum = enhanced_spectrum[:, :, 0, :]
        enhanced_frame = (
            torch.matmul(enhanced_spectrum[..., 0], self.synthesis_real)
            + torch.matmul(enhanced_spectrum[..., 1], self.synthesis_imag)
        ) * self.window

        enhanced = enhanced_frame[:, :HOP_SIZE] + synthesis_cache
        synthesis_cache = enhanced_frame[:, HOP_SIZE:]
        analysis_cache = speech
        return (
            enhanced,
            conv_cache,
            tra_cache,
            inter_cache,
            analysis_cache,
            synthesis_cache,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Export streaming GTCRN with waveform I/O")
    parser.add_argument("--source-dir", default="debug/gtcrn")
    parser.add_argument(
        "--checkpoint",
        default="debug/gtcrn/checkpoints/model_trained_on_dns3.tar",
    )
    parser.add_argument("--output", default="public/models/gtcrn.onnx")
    parser.add_argument("--verify-frames", type=int, default=20)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument(
        "--quantize",
        type=int,
        default=1,
        help="Save dynamic int8 model next to ONNX using .onnx -> .int8.onnx.",
    )
    return parser.parse_args()


def load_model(source_dir: str, checkpoint: str):
    stream_dir = Path(source_dir).resolve() / "stream"
    sys.path.insert(0, str(stream_dir))
    from gtcrn import GTCRN
    from gtcrn_stream import StreamGTCRN
    from modules.convert import convert_to_stream

    offline_model = GTCRN().cpu().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    offline_model.load_state_dict(state["model"])

    stream_model = StreamGTCRN().cpu().eval()
    convert_to_stream(stream_model, offline_model)
    return StreamingWaveformGTCRN(stream_model).cpu().eval()


def initial_inputs(batch_size: int = 1):
    return (
        torch.zeros(batch_size, HOP_SIZE, dtype=torch.float32),
        torch.zeros(2, batch_size, 16, 16, 33, dtype=torch.float32),
        torch.zeros(2, 3, 1, batch_size, 16, dtype=torch.float32),
        torch.zeros(2, 1, batch_size * 33, 16, dtype=torch.float32),
        torch.zeros(batch_size, HOP_SIZE, dtype=torch.float32),
        torch.zeros(batch_size, HOP_SIZE, dtype=torch.float32),
    )


def add_metadata(output_path: str):
    model = onnx.load(output_path)
    metadata = {
        "model_type": "gtcrn_denoise",
        "sample_rate": str(SAMPLE_RATE),
        "frame_length": str(NFFT),
        "frame_shift": str(HOP_SIZE),
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
    torch.manual_seed(20260723)
    batch_size = 2
    waveform = torch.rand(batch_size, num_frames * HOP_SIZE, dtype=torch.float32) * 2.0 - 1.0
    torch_state = list(initial_inputs(batch_size)[1:])
    ort_state = [value.numpy().copy() for value in initial_inputs(batch_size)[1:]]
    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])

    waveform_diff = 0.0
    cache_diff = 0.0
    for offset in range(0, waveform.shape[1], HOP_SIZE):
        speech = waveform[:, offset : offset + HOP_SIZE]
        with torch.no_grad():
            torch_outputs = model(speech, *torch_state)
        feeds = {
            "speech": speech.numpy(),
            "conv_cache": ort_state[0],
            "tra_cache": ort_state[1],
            "inter_cache": ort_state[2],
            "analysis_cache": ort_state[3],
            "synthesis_cache": ort_state[4],
        }
        ort_outputs = session.run(None, feeds)
        waveform_diff = max(
            waveform_diff,
            float(np.max(np.abs(torch_outputs[0].numpy() - ort_outputs[0]))),
        )
        for torch_cache, ort_cache in zip(torch_outputs[1:], ort_outputs[1:]):
            cache_diff = max(
                cache_diff,
                float(np.max(np.abs(torch_cache.numpy() - ort_cache))),
            )
        torch_state = [value.detach().clone() for value in torch_outputs[1:]]
        ort_state = [value.copy() for value in ort_outputs[1:]]

    print(f"PyTorch vs ONNX max abs diff: waveform={waveform_diff:.8g}, caches={cache_diff:.8g}")
    if waveform_diff >= 1e-4 or cache_diff >= 1e-4:
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


def convert_fp16_onnx_model(input_path: str, output_path: str):
    model = onnx.load(input_path)
    model = float16.convert_float_to_float16(model)
    onnx.save(model, output_path)
    add_metadata(output_path)
    print(f"Float16 model saved to: {output_path}")


def verify_fp16(output_path: str):
    model = onnx.load(output_path)
    onnx.checker.check_model(model)
    graph_values = [*model.graph.input, *model.graph.output]
    if any(value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT16 for value in graph_values):
        raise RuntimeError("Float16 ONNX model has non-float16 inputs or outputs")
    print("Float16 ONNX input/output dtype verification passed")


def export(model: nn.Module, output_path: str, quantize: bool):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy_inputs = initial_inputs()
    torch.onnx.export(
        model,
        dummy_inputs,
        str(output),
        input_names=[
            "speech",
            "conv_cache",
            "tra_cache",
            "inter_cache",
            "analysis_cache",
            "synthesis_cache",
        ],
        output_names=[
            "enhanced",
            "conv_cache_out",
            "tra_cache_out",
            "inter_cache_out",
            "analysis_cache_out",
            "synthesis_cache_out",
        ],
        # The streaming frame length remains fixed at HOP_SIZE. Only the batch
        # dimension is dynamic, and each cache must use the matching batch size.
        # inter_cache flattens batch and the fixed 33 frequency bins into axis 2,
        # so its dynamic dimension has size batch_size * 33.
        dynamic_axes={
            "speech": {0: "batch"},
            "conv_cache": {1: "batch"},
            "tra_cache": {3: "batch"},
            "inter_cache": {2: "batch_frequency"},
            "analysis_cache": {0: "batch"},
            "synthesis_cache": {0: "batch"},
            "enhanced": {0: "batch"},
            "conv_cache_out": {1: "batch"},
            "tra_cache_out": {3: "batch"},
            "inter_cache_out": {2: "batch_frequency"},
            "analysis_cache_out": {0: "batch"},
            "synthesis_cache_out": {0: "batch"},
        },
        opset_version=OPSET_VERSION,
        dynamo=False,
    )
    add_metadata(str(output))
    print(f"Exported GTCRN waveform model to: {output}")

    if quantize:
        quantized_path = str(output).replace(".onnx", ".int8.onnx")
        if quantized_path == str(output):
            raise ValueError(f"ONNX path must end with .onnx for int8 output: {output}")
        quantize_onnx_model(str(output), quantized_path)
        add_metadata(quantized_path)
        size = os.path.getsize(quantized_path)
        print(f"Int8 file size: {size:,} bytes ({size / 1024 / 1024:.2f} MB)")

    fp16_path = str(output).replace(".onnx", ".fp16.onnx")
    if fp16_path == str(output):
        raise ValueError(f"ONNX path must end with .onnx for FP16 output: {output}")
    convert_fp16_onnx_model(str(output), fp16_path)
    verify_fp16(fp16_path)


def main():
    args = parse_args()
    if args.verify_frames <= 0:
        raise ValueError("--verify-frames must be greater than zero")
    model = load_model(args.source_dir, args.checkpoint)
    export(model, args.output, quantize=bool(args.quantize))
    if not args.no_verify:
        verify(model, args.output, args.verify_frames)


if __name__ == "__main__":
    main()
