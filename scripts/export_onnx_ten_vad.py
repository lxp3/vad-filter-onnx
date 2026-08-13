#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import re
import sys

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxsim import simplify

opset_version = 18


"""
TEN VAD streaming input notes
=============================

This exporter builds an ONNX graph whose input is waveform samples instead of
pre-computed features. The graph itself does:

    normalized float waveform -> PCM-scale waveform -> pre-emphasis -> Hann
    window -> 1024-pt rFFT -> 40-bin log mel -> per-dim mean/var
    normalization -> 3-frame context stack -> separable conv stack -> 2x LSTM
    -> MLP -> sigmoid -> speech probability

There is no torch checkpoint upstream
--------------------------------------
TEN VAD ships only a TensorFlow-exported ONNX file
(src/onnx_model/ten-vad.onnx, tf2onnx 1.10.0, opset 9, Keras model
"vad_model") plus C/C++ feature-extraction sources and prebuilt shared
libraries. There is no .pt/.pth/.ckpt anywhere. This exporter therefore reads
the 19 weight initializers out of that upstream ONNX graph, loads them into an
equivalent torch nn.Module, and re-exports with the frontend traced in.

Audio format
------------
- Sample rate: 16000 Hz only.
- Input tensor name: samples.
- Input tensor shape: [1, 768].
- Input dtype: float32.
- Input scale: normalized float in [-1, 1]. The graph multiplies by 32768
  internally, because upstream TEN VAD is an int16 pipeline and its feature
  normalization statistics were fitted on PCM-scale magnitudes.

Feature frame geometry
-----------------------
- Analysis window: 768 samples (48 ms), Hann.
- Hop / frame shift: 256 samples (16 ms).
- FFT size: 1024 (the 768-sample windowed frame is zero-padded on the right).
- Mel bands: 40, plus 1 pitch dim = 41 features per frame.
- Context: 3 frames. The current frame's 41 features are concatenated after
  the 2 frames carried in cache_features, giving the [1, 3, 41] conv input.
- Pre-emphasis: 0.97, applied continuously across the frame.

Mel filterbank correctness
---------------------------
The filterbank is built here from upstream's own formula in src/aed.cc:
HTK-style 2595/700 mel warping, 0-8000 Hz, unnormalized triangles over
INTEGER bin edges computed as (fft_size + 1) * hz / sample_rate.

This matters. The previously checked-in public/models/ten_vad.onnx embedded a
librosa Slaney-normalized filterbank (htk=False, fmin=20, fmax=8000, matching
to 9.3e-10) instead. That is a different frontend from the one the weights
were trained with, and it measurably hurts accuracy: scoring the upstream
network against the 30 labeled files in TEN VAD's own testset/*.scv, frame
accuracy is 0.9001 with the upstream filterbank versus 0.8491 with the
Slaney one. This exporter uses the upstream filterbank.

Pitch feature (dim 40)
-----------------------
- Upstream computes an LPCNet-derived F0 estimate: 18-band log energies ->
  DCT -> order-16 LPC (Levinson-Durbin) -> LPC inverse filtering with a 0.7
  pitch prefilter tap -> 5-section biquad IIR decimation 16k->4k ->
  normalized cross-correlation over lags 8..64 -> Viterbi-style max-path
  tracking -> weighted linear regression. That is stateful, data-dependent
  control flow and is not sanely expressible as streaming ONNX ops.
- So pitch is a graph INPUT here, not a computed tensor: the C++ side owns it
  (vad-filter-onnx/utils/pitch-estimator.{h,cc}) and feeds one value per
  frame. Units are Hz, and 0.0 means "unvoiced" - the same convention
  upstream uses before normalization.
- Normalization for this dim is (mean 92.35690307617, std 115.2136917114),
  i.e. unscaled frequency, confirming the Hz convention.
- Note the previous checked-in model hardcoded this dim to a zeros
  initializer, permanently biasing the input by -0.80 sigma.

Recurrent state
---------------
- Two forward LSTMs, hidden size 64 each. h/c are graph inputs and outputs.
- cache_features [1, 2, 41] carries the previous 2 feature frames.
- The classifier sees concat(lstm2_out, lstm1_out) = 128 dims, i.e. there is
  a skip connection around the second LSTM.
- All five state tensors must be zero-initialized at stream start and then
  threaded call to call.

Inputs
------
- samples        [1, 768]   float32, normalized float waveform
- pitch          [1, 1]     float32, F0 in Hz (0.0 = unvoiced)
- h1, c1         [1, 64]    float32, LSTM1 state
- h2, c2         [1, 64]    float32, LSTM2 state
- cache_features [1, 2, 41] float32, previous 2 normalized feature frames

Outputs
-------
- prob                [1, 1]     float32 speech probability
- h1_out, c1_out      [1, 64]
- h2_out, c2_out      [1, 64]
- cache_features_out  [1, 2, 41]

Chunk length requirements
--------------------------
- Exactly 768 samples per call; all shapes are static (no dynamic axes).
- The caller advances by 256 samples per call, so consecutive windows
  overlap by 512 samples.

Post-processing
----------------
- This ONNX model outputs a raw per-frame speech probability only.
- Thresholding, sliding-window smoothing, min-speech/min-silence duration and
  segment timestamps stay outside the model; the C++ side handles them in
  VadModel::update_frame_state, shared with every other VAD in this project.
- Upstream applies a 1-frame lookahead, so probability t nominally describes
  audio at t + 16 ms.
"""

SAMPLE_RATE = 16000
FRAME_LENGTH = 768
FRAME_SHIFT = 256
N_FFT = 1024
N_MELS = 40
FEAT_DIM = N_MELS + 1
CONTEXT = 3
HIDDEN = 64
PREEMPH = 0.97
LOG_EPS = 1e-20
PCM_SCALE = 32768.0

# Upstream src/coeff.h, relative to the TEN VAD repo root.
COEFF_HEADER = "src/coeff.h"
UPSTREAM_ONNX = "src/onnx_model/ten-vad.onnx"


def get_args():
    parser = argparse.ArgumentParser(
        description="Export TEN VAD to ONNX with the mel frontend baked in."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data/user/lxp/llm/downloads/models/TEN-framework/ten-vad",
        help="Path to the TEN VAD repo/snapshot root (containing src/).",
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default=None,
        help="Output ONNX path. Defaults to <model-dir>/ten_vad.onnx.",
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


def parse_c_float_array(text, symbol):
    """Pull a flat float array out of a C header by symbol name."""
    start = text.index(symbol)
    open_brace = text.index("{", start)
    close_brace = text.index("}", open_brace)
    body = text[open_brace + 1 : close_brace]
    values = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", body)
    return np.array([float(v) for v in values], dtype=np.float32)


def build_mel_filterbank():
    """
    Reproduce upstream's filterbank from src/aed.cc exactly: HTK 2595/700
    warping, 0-8000 Hz, unnormalized triangles, integer bin edges using
    (fft_size + 1) in the hz->bin mapping.
    """
    low_mel = 2595.0 * math.log10(1.0 + 0.0 / 700.0)
    high_mel = 2595.0 * math.log10(1.0 + 8000.0 / 700.0)

    edges = []
    for i in range(N_MELS + 2):
        mel_point = i * (high_mel - low_mel) / (N_MELS + 1.0) + low_mel
        hz_point = 700.0 * (10.0 ** (mel_point / 2595.0) - 1.0)
        edges.append(int((N_FFT + 1.0) * hz_point / SAMPLE_RATE))

    for i in range(1, len(edges)):
        if edges[i] == edges[i - 1]:
            raise ValueError(f"Degenerate mel bin edge at {i}: {edges[i]}")

    n_bins = N_FFT // 2 + 1
    fb = np.zeros((N_MELS, n_bins), dtype=np.float32)
    for j in range(N_MELS):
        left, center, right = edges[j], edges[j + 1], edges[j + 2]
        for i in range(left, center):
            fb[j, i] = (i - left) / float(center - left)
        for i in range(center, right):
            fb[j, i] = (right - i) / float(right - center)
    return fb


class TenVadFrontend(nn.Module):
    """
    ONNX-exportable reimplementation of TEN VAD's feature extraction
    (src/aed.cc + src/stft.cc), for the fixed 16 kHz / 768-sample /
    256-hop configuration.

    Everything is precomputed into buffers so the traced graph contains only
    plain ops. The 768-sample frame is windowed then right-zero-padded to
    1024 before the rFFT, matching upstream's zero-padded FFT.
    """

    def __init__(self, mel_filters, window, norm_mean, norm_std):
        super().__init__()
        # Store the filterbank already transposed to [n_bins, n_mels]. Doing
        # the transpose here rather than in forward() keeps the initializer's
        # name in the exported graph, which is what the int8 exclusion list
        # matches on; a traced .t() gets folded into an anonymous "t"
        # initializer that then slips through and gets quantized, wrecking the
        # frontend.
        self.register_buffer(
            "mel_filters_t", torch.from_numpy(np.ascontiguousarray(mel_filters.T))
        )
        self.register_buffer("window", torch.from_numpy(window.copy()))
        self.register_buffer("norm_mean", torch.from_numpy(norm_mean.copy()))
        # Upstream divides by (std + EPS); fold to a reciprocal so the graph
        # uses a Mul instead of a Div.
        self.register_buffer(
            "norm_inv_std", torch.from_numpy((1.0 / (norm_std + LOG_EPS)).astype(np.float32))
        )
        # log(power / 32768^2) == log(power) - log(32768^2); folding the
        # division into a subtraction after the log keeps the mel matmul in
        # PCM scale, exactly as upstream computes it.
        self.register_buffer(
            "log_power_offset", torch.tensor(math.log(PCM_SCALE * PCM_SCALE), dtype=torch.float32)
        )

    def forward(self, samples, pitch):
        # samples: [1, 768] normalized float -> PCM scale.
        x = samples * PCM_SCALE

        # Continuous pre-emphasis. Upstream carries the last sample of the
        # previous hop as state; within a 768-sample window that only affects
        # sample 0, whose windowed value is 0.0 (Hann starts at zero), so
        # replicating sample 0 here is numerically equivalent.
        shifted = torch.cat([x[:, :1], x[:, :-1]], dim=1)
        x = x - PREEMPH * shifted

        x = x * self.window
        x = F.pad(x, (0, N_FFT - FRAME_LENGTH), mode="constant", value=0.0)

        spec = torch.fft.rfft(x, n=N_FFT, dim=1)
        power = spec.real.pow(2) + spec.imag.pow(2)  # [1, 513]

        mel = torch.matmul(power, self.mel_filters_t)  # [1, 40]
        log_mel = torch.log(mel + LOG_EPS) - self.log_power_offset

        # Pitch occupies dim 40, in Hz, 0.0 when unvoiced.
        feat = torch.cat([log_mel, pitch], dim=1)  # [1, 41]
        feat = (feat - self.norm_mean) * self.norm_inv_std
        return feat.unsqueeze(1)  # [1, 1, 41]


class TenVadNet(nn.Module):
    """
    Torch equivalent of the upstream Keras "vad_model" graph:

        [1, 3, 41] -> SeparableConv2d(3x3 -> 16) ReLU
                   -> MaxPool k[1,3] s[1,2]
                   -> SeparableConv2d(s2, pad(0,1,0,1)) ReLU
                   -> SeparableConv2d(s2, pad(0,0,0,1)) ReLU
                   -> reshape [1, 80]
                   -> LSTMCell(80 -> 64) -> LSTMCell(64 -> 64)
                   -> concat(lstm2, lstm1) = 128
                   -> Dense(128 -> 32) ReLU -> Dense(32 -> 1) -> Sigmoid
    """

    def __init__(self):
        super().__init__()
        self.conv1_dw = nn.Conv2d(1, 1, kernel_size=3, bias=False)
        self.conv1_pw = nn.Conv2d(1, 16, kernel_size=1, bias=True)
        self.conv2_dw = nn.Conv2d(16, 16, kernel_size=(1, 3), stride=2, groups=16, bias=False)
        self.conv2_pw = nn.Conv2d(16, 16, kernel_size=1, bias=True)
        self.conv3_dw = nn.Conv2d(16, 16, kernel_size=(1, 3), stride=2, groups=16, bias=False)
        self.conv3_pw = nn.Conv2d(16, 16, kernel_size=1, bias=True)
        self.lstm1 = nn.LSTMCell(80, HIDDEN)
        self.lstm2 = nn.LSTMCell(HIDDEN, HIDDEN)
        self.dense3 = nn.Linear(128, 32)
        self.dense5 = nn.Linear(32, 1)

    def forward(self, feats, h1, c1, h2, c2):
        # feats: [1, 3, 41]
        x = feats.unsqueeze(1)  # [1, 1, 3, 41]
        x = self.conv1_pw(self.conv1_dw(x))
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(1, 3), stride=(1, 2))

        # Upstream ONNX pads are [H_begin, W_begin, H_end, W_end] = [0,1,0,1]
        # here, i.e. one column on each side of the width axis (height is
        # already 1 after the first conv). F.pad order is
        # (left, right, top, bottom).
        x = F.pad(x, (1, 1, 0, 0))
        x = F.relu(self.conv2_pw(self.conv2_dw(x)))

        # Second strided conv pads [0,0,0,1]: one column on the right only.
        x = F.pad(x, (0, 1, 0, 0))
        x = F.relu(self.conv3_pw(self.conv3_dw(x)))

        x = x.reshape(1, 16, 5).transpose(1, 2).reshape(1, 80)

        h1_out, c1_out = self.lstm1(x, (h1, c1))
        h2_out, c2_out = self.lstm2(h1_out, (h2, c2))

        # Skip connection: the classifier sees both LSTM outputs.
        joint = torch.cat([h2_out, h1_out], dim=-1)  # [1, 128]
        prob = torch.sigmoid(self.dense5(F.relu(self.dense3(joint))))
        return prob, h1_out, c1_out, h2_out, c2_out


class TenVadWrapper(nn.Module):
    def __init__(self, frontend, net):
        super().__init__()
        self.frontend = frontend
        self.net = net

    def forward(self, samples, pitch, h1, c1, h2, c2, cache_features):
        feat = self.frontend(samples, pitch)  # [1, 1, 41]
        stacked = torch.cat([cache_features, feat], dim=1)  # [1, 3, 41]
        prob, h1_out, c1_out, h2_out, c2_out = self.net(stacked, h1, c1, h2, c2)
        cache_out = stacked[:, 1:, :]  # keep the last 2 frames
        return prob, h1_out, c1_out, h2_out, c2_out, cache_out


# ONNX LSTM packs gates as [i, o, f, c]; torch uses [i, f, c, o]. Reordering
# the ONNX blocks by these indices yields the torch layout. Verified to be
# bit-exact (max abs diff 0.0) against a known-good torch-ordered export.
ONNX_TO_TORCH_GATES = (0, 2, 3, 1)

# tf2onnx keeps the original Keras variable names for the trained weights, so
# these are stable identifiers. Only the depthwise kernels get folded into
# opaque const_fold_opt__NNN names, and those are matched by shape (each is
# unique among the remaining candidates once the others are claimed).
KERAS_PREFIX = "StatefulPartitionedCall/vad_model/"
UPSTREAM_NAMES = {
    "conv1_pw.weight": KERAS_PREFIX + "separable_conv2d/separable_conv2d/ReadVariableOp_1:0",
    "conv1_pw.bias": KERAS_PREFIX + "separable_conv2d/BiasAdd/ReadVariableOp:0",
    "conv2_pw.weight": KERAS_PREFIX + "separable_conv1d/ExpandDims_2:0",
    "conv2_pw.bias": KERAS_PREFIX + "separable_conv1d/BiasAdd/ReadVariableOp:0",
    "conv3_pw.weight": KERAS_PREFIX + "separable_conv1d_1/ExpandDims_2:0",
    "conv3_pw.bias": KERAS_PREFIX + "separable_conv1d_1/BiasAdd/ReadVariableOp:0",
    "dense3.weight": KERAS_PREFIX + "dense_3/Tensordot/ReadVariableOp:0",
    "dense3.bias": KERAS_PREFIX + "dense_3/BiasAdd/ReadVariableOp:0",
    "dense5.weight": KERAS_PREFIX + "dense_5/Tensordot/ReadVariableOp:0",
    "dense5.bias": KERAS_PREFIX + "dense_5/BiasAdd/ReadVariableOp:0",
}

# Depthwise kernels, in graph order, matched by shape.
UPSTREAM_DEPTHWISE = [
    ("conv1_dw.weight", (1, 1, 3, 3)),
    ("conv2_dw.weight", (16, 1, 1, 3)),
    ("conv3_dw.weight", (16, 1, 1, 3)),
]

# LSTM W/R/B initializer names, in graph order.
UPSTREAM_LSTM = [
    ("lstm1", "W0__70", "R0__71", "B0__72"),
    ("lstm2", "W0__99", "R0__100", "B0__101"),
]


def load_upstream_weights(net, upstream_onnx_path):
    """
    Copy the trained weights out of upstream's tf2onnx graph into the torch
    module, applying the Keras -> torch layout conversions:
      - Dense kernels are stored transposed relative to nn.Linear.
      - LSTM gate blocks are reordered from ONNX [i,o,f,c] to torch [i,f,c,o].
      - Keras keeps one LSTM bias; ONNX splits it into Wb|Rb, so sum them and
        put the total on bias_ih with bias_hh zeroed.
    """
    model = onnx.load(upstream_onnx_path)
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}

    def reorder_gates(arr):
        blocks = [arr[HIDDEN * i : HIDDEN * (i + 1)] for i in ONNX_TO_TORCH_GATES]
        return np.concatenate(blocks, axis=0)

    with torch.no_grad():
        for attr, onnx_name in UPSTREAM_NAMES.items():
            if onnx_name not in inits:
                raise KeyError(f"Upstream ONNX is missing expected initializer: {onnx_name}")
            arr = inits[onnx_name]
            module_name, param_name = attr.split(".")
            param = getattr(getattr(net, module_name), param_name)
            if arr.ndim == 2 and tuple(arr.T.shape) == tuple(param.shape):
                arr = arr.T  # Keras Dense kernel is [in, out]
            if tuple(arr.shape) != tuple(param.shape):
                raise ValueError(
                    f"Shape mismatch for {attr}: upstream {arr.shape} vs torch {tuple(param.shape)}"
                )
            param.copy_(torch.from_numpy(np.ascontiguousarray(arr)))

        # Depthwise kernels: claim shape matches in graph order, without
        # reusing an initializer already bound above.
        claimed = set(UPSTREAM_NAMES.values())
        for attr, shape in UPSTREAM_DEPTHWISE:
            candidates = [
                name
                for name, arr in inits.items()
                if name not in claimed and tuple(arr.shape) == shape
            ]
            if not candidates:
                raise ValueError(f"No upstream initializer with shape {shape} for {attr}")
            name = candidates[0]
            claimed.add(name)
            module_name, param_name = attr.split(".")
            getattr(getattr(net, module_name), param_name).copy_(
                torch.from_numpy(np.ascontiguousarray(inits[name]))
            )

        for attr, w_name, r_name, b_name in UPSTREAM_LSTM:
            cell = getattr(net, attr)
            cell.weight_ih.copy_(torch.from_numpy(reorder_gates(inits[w_name][0]).copy()))
            cell.weight_hh.copy_(torch.from_numpy(reorder_gates(inits[r_name][0]).copy()))
            bias = inits[b_name][0]
            total = reorder_gates(bias[: 4 * HIDDEN]) + reorder_gates(bias[4 * HIDDEN :])
            cell.bias_ih.copy_(torch.from_numpy(total.copy()))
            cell.bias_hh.zero_()

    print(f"Loaded upstream weights from: {upstream_onnx_path}")


def add_metadata_to_onnx(onnx_path, metadata_dict):
    model = onnx.load(onnx_path)
    del model.metadata_props[:]
    for key, value in metadata_dict.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)
    onnx.save(model, onnx_path)
    print(f"Added metadata: {metadata_dict}")


def inline_external_data(onnx_path):
    """Collapse any external weight sidecar back into the .onnx file."""
    model = onnx.load(onnx_path, load_external_data=True)
    for init in model.graph.initializer:
        init.ClearField("data_location")
        del init.external_data[:]
    onnx.save(model, onnx_path, save_as_external_data=False)

    sidecar = onnx_path + ".data"
    if os.path.exists(sidecar):
        os.remove(sidecar)
        print(f"Inlined external data and removed {os.path.basename(sidecar)}")


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
        "mel",
        "window",
        "norm_",
        "dft",
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


INPUT_NAMES = ["samples", "pitch", "h1", "c1", "h2", "c2", "cache_features"]
OUTPUT_NAMES = ["prob", "h1_out", "c1_out", "h2_out", "c2_out", "cache_features_out"]


def make_dummy_inputs():
    torch.manual_seed(0)
    return (
        torch.randn(1, FRAME_LENGTH, dtype=torch.float32).clamp(-1.0, 1.0),
        torch.full((1, 1), 137.0, dtype=torch.float32),
        torch.zeros(1, HIDDEN, dtype=torch.float32),
        torch.zeros(1, HIDDEN, dtype=torch.float32),
        torch.zeros(1, HIDDEN, dtype=torch.float32),
        torch.zeros(1, HIDDEN, dtype=torch.float32),
        torch.zeros(1, CONTEXT - 1, FEAT_DIM, dtype=torch.float32),
    )


def verify_onnx(onnx_path, wrapper, dummy_inputs):
    import onnxruntime as ort

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified OK")

    with torch.no_grad():
        torch_outs = wrapper(*dummy_inputs)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {
        name: tensor.cpu().numpy() for name, tensor in zip(INPUT_NAMES, dummy_inputs)
    }
    ort_outs = session.run(None, ort_inputs)

    worst = 0.0
    for name, ort_out, torch_out in zip(OUTPUT_NAMES, ort_outs, torch_outs):
        diff = float(np.max(np.abs(ort_out - torch_out.cpu().numpy())))
        worst = max(worst, diff)
        print(f"  PyTorch vs ONNX max abs diff: {name}={diff:.8f}")
    assert worst < 1e-4, f"output diff too large: {worst}"


def verify_against_upstream(wrapper, upstream_onnx_path):
    """
    Cross-check the whole pipeline against upstream's own graph. The upstream
    model takes the pre-stacked [1, 3, 41] feature tensor, so drive both with
    the same features and compare probabilities frame by frame.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(upstream_onnx_path, providers=["CPUExecutionProvider"])
    up_inputs = [i.name for i in session.get_inputs()]

    torch.manual_seed(1)
    num_frames = 40
    audio = (torch.randn(1, FRAME_LENGTH + FRAME_SHIFT * (num_frames - 1)) * 0.2).clamp(-1, 1)
    pitches = torch.where(
        torch.arange(num_frames) % 3 == 0,
        torch.zeros(num_frames),
        torch.linspace(80.0, 300.0, num_frames),
    )

    h1 = c1 = h2 = c2 = torch.zeros(1, HIDDEN)
    cache = torch.zeros(1, CONTEXT - 1, FEAT_DIM)
    up_state = {name: np.zeros((1, HIDDEN), dtype=np.float32) for name in up_inputs[1:]}
    up_ctx = np.zeros((1, CONTEXT, FEAT_DIM), dtype=np.float32)

    worst = 0.0
    for t in range(num_frames):
        frame = audio[:, t * FRAME_SHIFT : t * FRAME_SHIFT + FRAME_LENGTH]
        pitch = pitches[t].reshape(1, 1)
        with torch.no_grad():
            feat = wrapper.frontend(frame, pitch)
            prob, h1, c1, h2, c2, cache = wrapper(frame, pitch, h1, c1, h2, c2, cache)

        # Feed upstream the identical feature stack.
        up_ctx = np.concatenate([up_ctx[:, 1:, :], feat.numpy()], axis=1)
        up_out = session.run(None, {up_inputs[0]: up_ctx.astype(np.float32), **up_state})
        for i, name in enumerate(up_inputs[1:]):
            up_state[name] = up_out[i + 1]

        worst = max(worst, abs(float(np.ravel(up_out[0])[0]) - float(prob.item())))

    print(f"Torch vs upstream ONNX max abs prob diff over {num_frames} frames: {worst:.8f}")
    assert worst < 1e-4, f"upstream prob diff too large: {worst}"


def export_onnx(model_path, output_path, opset, skip_simplify, verify, quantize):
    upstream_onnx = os.path.join(model_path, UPSTREAM_ONNX)
    coeff_header = os.path.join(model_path, COEFF_HEADER)
    for path in (upstream_onnx, coeff_header):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing required upstream file: {path}")

    print(f"Reading upstream coefficients from: {coeff_header}")
    header_text = open(coeff_header).read()
    window = parse_c_float_array(header_text, "AUP_AED_STFTWindow_Hann768")
    norm_mean = parse_c_float_array(header_text, "AUP_AED_FEATURE_MEANS")
    norm_std = parse_c_float_array(header_text, "AUP_AED_FEATURE_STDS")
    assert window.shape == (FRAME_LENGTH,), window.shape
    assert norm_mean.shape == (FEAT_DIM,), norm_mean.shape
    assert norm_std.shape == (FEAT_DIM,), norm_std.shape

    mel_filters = build_mel_filterbank()
    print(f"Built mel filterbank {mel_filters.shape} from upstream aed.cc formula")

    frontend = TenVadFrontend(mel_filters, window, norm_mean, norm_std)
    net = TenVadNet()

    load_upstream_weights(net, upstream_onnx)

    wrapper = TenVadWrapper(frontend, net).cpu().eval()

    dummy_inputs = make_dummy_inputs()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if verify:
        verify_against_upstream(wrapper, upstream_onnx)

    torch.onnx.export(
        wrapper,
        dummy_inputs,
        output_path,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        opset_version=opset,
        verbose=False,
        # The legacy TorchScript exporter cannot lower aten::fft_rfft; the
        # dynamo path emits a native ONNX DFT node for it.
        dynamo=True,
    )
    print(f"Exported model to: {output_path}")

    # The dynamo exporter writes weights to a sibling .onnx.data file. Fold
    # them back in so the model ships as one self-contained file, like every
    # other model in public/models/.
    inline_external_data(output_path)

    if not skip_simplify:
        simplify_onnx(output_path)

    metadata = {
        "model_type": "ten_vad",
        "sample_rate": SAMPLE_RATE,
        "frame_length": FRAME_LENGTH,
        "frame_shift": FRAME_SHIFT,
        "input_scale": "normalized_float",
        "pitch_unit": "hz",
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
        onnx_path = os.path.join(args.model_path, "ten_vad.onnx")
    export_onnx(
        model_path=args.model_path,
        output_path=onnx_path,
        opset=args.opset,
        skip_simplify=args.skip_simplify,
        verify=bool(args.verify),
        quantize=bool(args.quantize),
    )


if __name__ == "__main__":
    sys.exit(main())
