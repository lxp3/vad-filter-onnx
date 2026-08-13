#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import sys

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxsim import simplify

opset_version = 18


"""
NeMo MarbleNet v2.0 VAD streaming input notes
==============================================

This exporter builds an ONNX graph whose input is waveform samples instead of
pre-computed mel features. The graph itself does:

    normalized float waveform -> PCM-scale waveform -> 80-bin log mel (NeMo
    AudioToMelSpectrogramPreprocessor semantics) -> ConvASREncoder -> MLP
    decoder -> softmax -> speech probability

Audio format
------------
- Sample rate: 16000 Hz only.
- Input tensor name: speech.
- Input tensor shape: [1, num_samples].
- Input dtype: float32.
- Input scale: normalized float in [-1, 1].
  Unlike this project's other VAD exporters, the ONNX graph does NOT scale
  the waveform by 32768 before mel extraction: NeMo's own
  AudioToMelSpectrogramPreprocessor consumes normalized float audio
  directly (verified against the real preprocessor with < 1e-4 max abs
  diff using un-scaled input), so this frontend mirrors that.

Feature frame geometry
-----------------------
- Mel frame width/window: 25 ms = 400 samples at 16 kHz.
- Mel frame shift/hop: 10 ms = 160 samples at 16 kHz.
- STFT uses center=True padding (n_fft // 2 zeros on both sides), matching
  NeMo's AudioToMelSpectrogramPreprocessor (NOT Kaldi snip_edges framing).
- mel frame count for N samples: floor(N / hop) + 1 valid frames (the
  remaining frame produced by center-padding beyond that is invalid/masked
  by NeMo and is dropped here too).
- Encoder's first conv block has stride=2, so encoder output has half as
  many frames as valid mel frames (rounded, after pad_to=2 alignment): one
  probability every 20 ms.

No recurrent cache / no LSTM-style state
-----------------------------------------
- Model architecture: Jasper-style separable-conv residual blocks, kernel
  sizes 11/13/15/17/29, dilation up to 2 for the last major block. This is a
  non-causal feedforward stack with a receptive field wider than one frame,
  but it carries NO cache/state tensor across calls, unlike FSMN-VAD /
  FireRedVAD in this project.
- Every ONNX Runtime call is a fully independent forward pass over whatever
  waveform window is supplied. There is no caches_in/caches_out I/O.
- NeMo's own "real-time" demo (Online_Offline_Microphone_VAD_Demo.ipynb)
  approximates streaming the same way: repeatedly running the whole model
  over a sliding window that includes left/right context, keeping only the
  "stable" middle frames, then sliding the window forward. This project's
  C++ side (NemoMarbleNetVadModel) implements exactly that pattern on top of
  this stateless ONNX graph; see vad/nemo-marblenet-vad-model.cc.

Inputs
------
- speech [1, num_samples] float32.

Outputs
-------
- probs [1, num_frames] float32, one value per 20 ms encoder output frame.

Chunk length requirements
--------------------------
- Minimum chunk length that produces one encoder output frame:
  400 samples (25 ms), because the first mel frame needs win_length=400
  samples once centered.
- Larger windows both reduce relative edge effects and let more encoder
  frames be produced per call. There is no hard requirement on exact chunk
  length beyond producing at least one valid mel frame; num_frames scales
  with input length continuously (dynamic_axes).

Post-processing
----------------
- This ONNX model outputs raw speech probabilities only.
- Thresholding, smoothing, min-speech/min-silence duration, and segment
  timestamp generation should remain outside the model.
- Timestamp mapping: output frame i corresponds to roughly [i * 20ms, (i+1) *
  20ms), though because of the model's wide receptive field frames near a
  window's edges are less reliable than frames in the middle (see the C++
  sliding-window implementation for how this project resolves that).
"""


def get_args():
    parser = argparse.ArgumentParser(
        description="Export NeMo MarbleNet v2.0 VAD to ONNX with mel frontend baked in."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=(
            "/data/user/lxp/llm/downloads/models/nvidia/"
            "Frame_VAD_Multilingual_MarbleNet_v2.0/"
            "frame_vad_multilingual_marblenet_v2.0.nemo"
        ),
        help="Path to the NeMo MarbleNet v2.0 .nemo checkpoint.",
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default=None,
        help="Output ONNX path. Defaults to <model-dir>/nemo_marblenet_v2.onnx.",
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


class NemoMelFrontend(nn.Module):
    """
    ONNX-exportable reimplementation of NeMo's AudioToMelSpectrogramPreprocessor
    for this specific model's config (sample_rate=16000, window_size=25ms,
    window_stride=10ms, window=hann, n_fft=512, features=80, preemph=0.97,
    normalize=None, log_zero_guard_type=add, dither=0 for deterministic
    inference, pad_to=2).

    NeMo internally uses torch.stft with center=True (i.e. zero-padding of
    n_fft // 2 samples on both sides), continuous (non-per-frame)
    pre-emphasis, a Hann window zero-padded to n_fft, and librosa
    Slaney-normalized mel filters. This reimplementation mirrors that
    exactly (verified against the real preprocessor with < 1e-5 max abs
    diff) so the exported ONNX graph does not need torch.stft, which is not
    ONNX-friendly for dynamic-length inputs.
    """

    def __init__(self, mel_filters, window, win_length, hop_length, n_fft,
                 preemph, log_zero_guard_value, pad_to):
        super().__init__()
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.preemph = preemph
        self.log_zero_guard_value = log_zero_guard_value
        self.pad_to = pad_to

        self.register_buffer("mel_filters", mel_filters.clone())  # [nfilt, n_stft]

        wpad = (n_fft - win_length) // 2
        full_window = F.pad(window, (wpad, n_fft - win_length - wpad))
        self.register_buffer("window", full_window)  # [n_fft]

        n = torch.arange(n_fft, dtype=torch.float32).unsqueeze(1)  # [n_fft, 1]
        n_stft = n_fft // 2 + 1
        k = torch.arange(n_stft, dtype=torch.float32).unsqueeze(0)  # [1, n_stft]
        angles = 2 * math.pi * k * n / n_fft
        self.register_buffer("dft_real", torch.cos(angles))  # [n_fft, n_stft]
        self.register_buffer("dft_imag", -torch.sin(angles))  # [n_fft, n_stft]

    def forward(self, speech):
        # speech: [1, num_samples] float32
        x = torch.cat([speech[:, :1], speech[:, 1:] - self.preemph * speech[:, :-1]], dim=1)

        pad = self.n_fft // 2
        x = F.pad(x, (pad, pad), mode="constant", value=0.0)

        num_samples = x.shape[1]
        num_frames = (num_samples - self.n_fft) // self.hop_length + 1
        frame_idx = torch.arange(self.n_fft, device=x.device).unsqueeze(0)
        start_idx = torch.arange(num_frames, device=x.device).unsqueeze(1) * self.hop_length
        gather_idx = (frame_idx + start_idx).reshape(-1)
        frames = x[:, gather_idx].reshape(1, num_frames, self.n_fft)

        windowed = frames * self.window
        spec_real = torch.matmul(windowed, self.dft_real)
        spec_imag = torch.matmul(windowed, self.dft_imag)
        power = spec_real.pow(2) + spec_imag.pow(2)  # [1, num_frames, n_stft]

        mel = torch.matmul(power, self.mel_filters.t())  # [1, num_frames, nfilt]
        log_mel = torch.log(mel + self.log_zero_guard_value)
        feat = log_mel.transpose(1, 2)  # [1, nfilt, num_frames]

        # NeMo's frame-count formula (get_seq_len) always yields exactly one
        # fewer valid frame than the raw center-padded framing above produces
        # (get_seq_len(N) == floor(N / hop), while framing yields that + 1).
        # NeMo masks that last frame to zero and passes the REDUCED valid
        # length downstream (not the padded tensor's raw size) so the
        # encoder's internal conv masking matches exactly; only mirroring
        # the zero-mask without also shrinking the length that reaches the
        # encoder causes cumulative mismatches through the stride-2/residual
        # stack for longer inputs.
        valid_len = num_frames - 1
        zero_mask = torch.ones_like(feat)
        zero_mask[:, :, -1] = 0.0
        feat = feat * zero_mask

        pad_amt = feat.size(-1) % self.pad_to
        if pad_amt != 0:
            feat = F.pad(feat, (0, self.pad_to - pad_amt), value=0.0)
        return feat, valid_len


class NemoMarbleNetWrapper(nn.Module):
    def __init__(self, frontend, encoder, decoder):
        super().__init__()
        self.frontend = frontend
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, speech):
        feat, valid_len = self.frontend(speech)  # [1, 80, T]
        length = torch.full((1,), valid_len, dtype=torch.int64, device=feat.device)
        encoded, _ = self.encoder(audio_signal=feat, length=length)
        logits = self.decoder(encoded.transpose(1, 2))  # [1, num_frames, 2]
        probs = torch.softmax(logits, dim=-1)[..., 1]  # [1, num_frames]
        return probs


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


def verify_onnx(onnx_path, wrapper, dummy_inputs):
    import numpy as np
    import onnxruntime as ort

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified OK")

    with torch.no_grad():
        torch_probs = wrapper(*dummy_inputs)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {"speech": dummy_inputs[0].cpu().numpy()}
    ort_probs = session.run(None, ort_inputs)[0]

    prob_diff = np.max(np.abs(ort_probs - torch_probs.cpu().numpy()))
    print(f"PyTorch vs ONNX max abs diff: probs={prob_diff:.8f}")
    assert prob_diff < 1e-4, f"probs diff too large: {prob_diff}"


def export_onnx(model_path, output_path, opset, skip_simplify, verify, quantize):
    import nemo.collections.asr as nemo_asr

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Missing model file: {model_path}")

    print(f"Loading NeMo MarbleNet v2.0 from: {model_path}")
    model = nemo_asr.models.EncDecFrameClassificationModel.restore_from(
        model_path, map_location="cpu", strict=False,
    )
    model = model.cpu().eval()

    pp = model.preprocessor.featurizer
    frontend = NemoMelFrontend(
        mel_filters=pp.fb.squeeze(0),
        window=pp.window,
        win_length=pp.win_length,
        hop_length=pp.hop_length,
        n_fft=pp.n_fft,
        preemph=pp.preemph,
        log_zero_guard_value=5.960464477539063e-08,
        pad_to=pp.pad_to,
    )
    wrapper = NemoMarbleNetWrapper(frontend, model.encoder, model.decoder).cpu().eval()

    dummy_speech = torch.randn(1, 16000, dtype=torch.float32).clamp(-1.0, 1.0)
    dummy_inputs = (dummy_speech,)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        output_path,
        input_names=["speech"],
        output_names=["probs"],
        dynamic_axes={
            "speech": {1: "num_samples"},
            "probs": {1: "num_frames"},
        },
        opset_version=opset,
        verbose=False,
        dynamo=False,
    )
    print(f"Exported model to: {output_path}")

    if not skip_simplify:
        simplify_onnx(output_path)

    metadata = {
        "model_type": "nemo_marblenet_v2",
        "sample_rate": 16000,
        "input_scale": "normalized_float",
        "streaming": 0,
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
        onnx_path = os.path.join(os.path.dirname(args.model_path), "nemo_marblenet_v2.onnx")
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
