#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import sys

import kaldiio
import onnx
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxsim import simplify

from filter_fbank import Filterbank

opset_version = 18


"""
FireRedVAD streaming input notes
================================

This exporter builds an ONNX graph whose input is waveform samples instead of
pre-computed fbank features. The graph itself does:

    normalized float waveform -> PCM-scale waveform -> 80-bin log fbank
    -> CMVN -> Stream-VAD DFSMN

Audio format
------------
- Sample rate: 16000 Hz only.
- Input tensor name: speech.
- Input tensor shape: [1, num_samples].
- Input dtype: float32.
- Input scale: normalized float, usually in [-1, 1].
  The ONNX graph multiplies waveform by 32768 before fbank, matching the
  FireRedVAD official feature amplitude and this project's FSMN-VAD export.

Feature frame geometry
----------------------
- Frame width/window: 25 ms = 400 samples at 16 kHz.
- Frame shift/hop: 10 ms = 160 samples at 16 kHz.
- snip_edges=True, so only complete 400-sample frames are emitted.
- For an input chunk with N samples, fbank frame count is:

      num_frames = floor((N - 400) / 160) + 1, when N >= 400

- The model has no LFR stage. One fbank frame corresponds to one VAD probability
  frame, i.e. one output every 10 ms.

Streaming cache
---------------
- The ONNX model exported here is the with-cache streaming variant.
- Inputs:
    speech     [1, num_samples]
    caches_in  [8, 1, 128, 19]
- Outputs:
    probs      [1, num_frames, 1]
    caches_out [8, 1, 128, 19]
- The 8 cache slices correspond to the first FSMN block plus the remaining
  7 DFSMN blocks. Each cache stores 19 lookback frames because FireRedVAD
  Stream-VAD uses N1=20, S1=1, so lookback_padding=(N1 - 1) * S1 = 19.
- For the first chunk, pass all-zero caches_in.
- For every later chunk, pass the previous call's caches_out as caches_in.
- Reset caches to zero when starting a new independent audio stream.
- The cache stores model history only. The caller still needs waveform overlap
  handling when splitting arbitrary sample packets into fbank-aligned chunks.

Chunk length requirements
-------------------------
- Minimum chunk length that produces one frame:

      400 samples = 25 ms

- A chunk aligned to K output frames should have:

      num_samples = 400 + (K - 1) * 160

  Examples:
    1 frame   -> 400 samples  -> 25 ms
    10 frames -> 1840 samples -> 115 ms
    100 frames -> 16240 samples -> 1015 ms

- Middle chunks are easiest to handle when their length follows the formula
  above. This avoids dropping a sub-frame tail inside the ONNX graph, because
  snip_edges=True ignores samples after the last complete frame.
- If the caller receives arbitrary streaming packet sizes, keep an external
  waveform remainder buffer:
    1. Append new normalized float samples to the buffer.
    2. Choose K frames to process.
    3. Feed exactly 400 + (K - 1) * 160 samples.
    4. Remove K * 160 samples from the buffer, leaving the last 240 samples of
       overlap plus any new partial-frame tail.

First chunk
-----------
- There is no special ONNX input flag for "first chunk".
- Use zero caches_in.
- The first chunk only needs to satisfy the same minimum 400-sample requirement.
- For lower scheduling overhead and stable post-processing, a practical first
  chunk is often 10 frames:

      1840 samples = 115 ms

- If using a 100 ms first packet (1600 samples), it produces:

      floor((1600 - 400) / 160) + 1 = 8 frames

Middle chunks
-------------
- Use the previous caches_out.
- Any K >= 1 is valid if the waveform chunk length is 400 + (K - 1) * 160.
- Recommended real-time chunk sizes are usually 5 to 20 frames:

      5 frames  -> 1040 samples -> 65 ms
      10 frames -> 1840 samples -> 115 ms
      20 frames -> 3440 samples -> 215 ms

- Smaller chunks reduce latency but increase ONNX Runtime call overhead.
- Larger chunks reduce overhead but increase detection latency.

Final chunk
-----------
- There is no explicit "is_final" input in this ONNX graph.
- Feed all remaining samples that can form at least one complete frame.
- If the final remainder is shorter than 400 samples, this model cannot produce
  a probability for it because fbank uses snip_edges=True.
- If the application must force a decision for a very short final tail, pad the
  waveform externally to at least 400 samples before the final call. Padding
  policy is application-specific; this exporter does not bake in zero/replicate
  padding for the tail.

Post-processing
---------------
- This ONNX model outputs raw speech probabilities only.
- Thresholding, smoothing, min-speech/min-silence duration, and segment timestamp
  generation should remain outside the model, matching FireRedVAD runtime logic.
- Timestamp mapping is direct: output frame i advances by 10 ms.
"""


def get_args():
    parser = argparse.ArgumentParser(
        description="Export FireRedVAD Stream-VAD to ONNX with fbank and CMVN."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/FireRedTeam/FireRedVAD/Stream-VAD",
        help="Path to FireRedVAD Stream-VAD model directory.",
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default=None,
        help="Output ONNX path. Defaults to <model-dir>/fireredvad_stream_vad.onnx.",
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


def povey_window(window_size, periodic=False, dtype=None):
    return torch.hann_window(window_size, periodic=periodic, dtype=dtype).pow(0.85)


def read_cmvn(cmvn_path):
    stats = kaldiio.load_mat(cmvn_path)
    assert stats.shape[0] == 2, f"Expected 2-row CMVN stats, got {stats.shape}"
    dim = stats.shape[-1] - 1
    count = stats[0, dim]
    assert count >= 1, f"Invalid CMVN count: {count}"

    means = []
    inverse_std_variances = []
    floor = 1e-20
    for d in range(dim):
        mean = stats[0, d] / count
        variance = stats[1, d] / count - mean * mean
        if variance < floor:
            variance = floor
        means.append(float(mean))
        inverse_std_variances.append(1.0 / math.sqrt(float(variance)))

    means = torch.tensor(means, dtype=torch.float32)
    inverse_std_variances = torch.tensor(inverse_std_variances, dtype=torch.float32)
    return means, inverse_std_variances


class FireRedWaveformFrontend(nn.Module):
    def __init__(self, cmvn_path):
        super().__init__()
        means, inverse_std_variances = read_cmvn(cmvn_path)
        self.register_buffer("cmvn_means", means.view(1, 1, -1))
        self.register_buffer("cmvn_istd", inverse_std_variances.view(1, 1, -1))
        self.fbank = Filterbank(
            sample_rate=16000,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            window_fn=povey_window,
            snip_edges=True,
        )

    def forward(self, speech):
        speech = speech * (1 << 15)
        feat = self.fbank(speech)
        feat = (feat - self.cmvn_means) * self.cmvn_istd
        return feat


class FireRedVadStreamingWithCache(nn.Module):
    def __init__(self, model, frontend, num_caches):
        super().__init__()
        self.model = model
        self.frontend = frontend
        self.num_caches = num_caches

    def forward(self, speech, caches_in):
        feat = self.frontend(speech)
        cache_list = [caches_in[i] for i in range(self.num_caches)]
        probs, new_caches = self.model(feat, caches=cache_list)
        return probs, torch.stack(new_caches)


def get_num_fsmn_blocks(model):
    return 1 + len(model.dfsmn.fsmns)


def get_cache_shape(model):
    fsmn = model.dfsmn.fsmn1
    channels = fsmn.lookback_filter.in_channels
    lookback_padding = fsmn.lookback_padding
    return (1, channels, lookback_padding)


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
    model = onnx.load(input_path)
    nodes_to_exclude = []
    preprocess_inits = []
    preprocess_keywords = (
        "frontend",
        "fbank",
        "cmvn",
        "mel_filters",
        "dft_matrix",
        "window",
    )

    for init in model.graph.initializer:
        if any(keyword in init.name.lower() for keyword in preprocess_keywords):
            preprocess_inits.append(init.name)

    for node in model.graph.node:
        node_name = node.name.lower()
        if node.op_type == "Conv":
            nodes_to_exclude.append(node.name)
            continue
        if any(inp in preprocess_inits for inp in node.input):
            nodes_to_exclude.append(node.name)
            continue
        if any(keyword in node_name for keyword in preprocess_keywords):
            nodes_to_exclude.append(node.name)

    nodes_to_exclude = sorted(set(nodes_to_exclude))
    print(f"Excluding {len(nodes_to_exclude)} nodes from int8 quantization")

    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=nodes_to_exclude,
        per_channel=False,
        reduce_range=False,
    )
    print(f"Quantized int8 model saved to: {output_path}")


def verify_onnx(onnx_path, wrapper, dummy_inputs):
    import numpy as np
    import onnxruntime as ort

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified OK")

    with torch.no_grad():
        torch_probs, torch_caches = wrapper(*dummy_inputs)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {
        "speech": dummy_inputs[0].cpu().numpy(),
        "caches_in": dummy_inputs[1].cpu().numpy(),
    }
    ort_probs, ort_caches = session.run(None, ort_inputs)

    prob_diff = np.max(np.abs(ort_probs - torch_probs.cpu().numpy()))
    cache_diff = np.max(np.abs(ort_caches - torch_caches.cpu().numpy()))
    print(f"PyTorch vs ONNX max abs diff: probs={prob_diff:.8f}, caches={cache_diff:.8f}")
    assert prob_diff < 1e-4, f"probs diff too large: {prob_diff}"
    assert cache_diff < 1e-3, f"caches diff too large: {cache_diff}"


def export_onnx(model_dir, output_path, opset, skip_simplify, verify, quantize):
    try:
        from fireredvad.core.detect_model import DetectModel
    except ImportError as exc:
        try:
            # FireRedVAD's package __init__ imports its native audio frontend,
            # which is not needed because this exporter provides its own FBank.
            from core.detect_model import DetectModel
        except ImportError:
            raise ImportError(
                "Could not import FireRedVAD DetectModel. Set PYTHONPATH to either "
                "the package root or its fireredvad directory."
            ) from exc

    model_path = os.path.join(model_dir, "model.pth.tar")
    cmvn_path = os.path.join(model_dir, "cmvn.ark")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not os.path.isfile(cmvn_path):
        raise FileNotFoundError(f"Missing CMVN file: {cmvn_path}")

    print(f"Loading FireRedVAD Stream-VAD from: {model_dir}")
    model = DetectModel.from_pretrained(model_dir).cpu().eval()
    frontend = FireRedWaveformFrontend(cmvn_path)
    num_caches = get_num_fsmn_blocks(model)
    cache_shape = get_cache_shape(model)
    wrapper = FireRedVadStreamingWithCache(model, frontend, num_caches).cpu().eval()

    dummy_speech = torch.randn(1, 16000, dtype=torch.float32).clamp(-1.0, 1.0)
    dummy_caches = torch.zeros(num_caches, *cache_shape, dtype=torch.float32)
    dummy_inputs = (dummy_speech, dummy_caches)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        output_path,
        input_names=["speech", "caches_in"],
        output_names=["probs", "caches_out"],
        dynamic_axes={
            "speech": {1: "num_samples"},
            "probs": {1: "num_frames"},
        },
        opset_version=opset,
        verbose=False,
        dynamo=False,
    )
    print(f"Exported streaming model to: {output_path}")
    print(f"Cache shape: [{num_caches}, {cache_shape[0]}, {cache_shape[1]}, {cache_shape[2]}]")

    if not skip_simplify:
        simplify_onnx(output_path)

    metadata = {
        "model_type": "firered_vad",
        "sample_rate": 16000,
        "input_scale": "normalized_float",
        "streaming": 1,
    }
    add_metadata_to_onnx(output_path, metadata)

    size = os.path.getsize(output_path)
    print(f"File size: {size:,} bytes ({size / 1024 / 1024:.2f} MB)")

    if verify:
        verify_onnx(output_path, wrapper, dummy_inputs)

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
    onnx_path = args.onnx_path
    if onnx_path is None:
        onnx_path = os.path.join(args.model_dir, "fireredvad_stream_vad.onnx")
    export_onnx(
        model_dir=args.model_dir,
        output_path=onnx_path,
        opset=args.opset,
        skip_simplify=args.skip_simplify,
        verify=bool(args.verify),
        quantize=bool(args.quantize),
    )


if __name__ == "__main__":
    sys.exit(main())
