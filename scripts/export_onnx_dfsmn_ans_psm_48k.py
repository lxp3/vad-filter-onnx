#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DFSMN-ANS-PSM 48k causal streaming denoise export notes
=========================================================

speech_dfsmn_ans_psm_48k_causal (modelscope) is a causal, streaming speech
enhancement model: a 9-layer causal Deep-FSMN mask network operating on
120-dim fbank features, producing a 961-bin real gain mask applied to the
STFT spectrum of the input waveform, followed by inverse STFT.

This script embeds the ENTIRE pipeline (fbank frontend, DFSMN mask network,
STFT analysis, mask application, ISTFT synthesis) as pure torch ops into a
single ONNX graph with an explicit per-hop streaming interface, following
the same state-tensor convention as this project's DPDFNet export
(scripts/export_onnx_dpdfnet.py):

Audio format
------------
- Sample rate: 48000 Hz only.
- Input tensor "speech": [1, hop_size] float32, normalized float in [-1, 1].
  hop_size = 960 samples (20 ms).
- Cache tensors round-trip unchanged in shape across calls:
    - "analysis_cache" / "analysis_cache_out": [1, hop_size] (previous 960
      raw samples; doubles as history for both the fbank frame and the STFT
      analysis frame, since both use a 1920-sample window / 960 hop).
    - "synthesis_cache" / "synthesis_cache_out": [1, hop_size] (ISTFT
      overlap-add carry).
    - "state_in" / "state_out": [fsmn_depth * (lorder - 1) * 256]
      flattened per-layer causal-conv history for the 9 UniDeepFsmn layers.
- Output "enhanced": [1, hop_size] float32.

Model internals
----------------
- fbank: 120 mel bins, 40 ms/20 ms frame length/shift, hamming window,
  dither disabled (0.0, vs. the upstream pipeline's dither=1.0) for
  deterministic ONNX export. Reimplemented by hand (DC removal,
  preemphasis, hamming window, zero-pad to the next power of two, DFT via
  a fixed cos/sin matmul, power spectrum, mel filterbank matmul, log) since
  torch.onnx.export (legacy TorchScript exporter, dynamo=False) cannot
  export aten::fft_rfft, which torchaudio.compliance.kaldi.fbank calls
  internally. The mel filterbank matrix itself is taken verbatim from
  torchaudio.compliance.kaldi.get_mel_banks (not retraced, just copied as a
  constant buffer) so the mel weights are bit-identical to Kaldi's.
- fbank is computed on the waveform scaled by 32768 (to match the int16
  range the upstream pipeline trains/runs on: raw_pcm * 32768), while the
  STFT/ISTFT path runs on the unscaled normalized-float waveform. The
  overall gain gets undone because the network only ever produces a
  dimensionless [0,1] sigmoid mask from the fbank branch, which is then
  applied multiplicatively to the (unscaled) STFT spectrum.
- DFSMN causal conv: each of the 9 UniDeepFsmn layers keeps a per-layer
  cache of its previous (lorder - 1) = 19 input frames (own weights are
  read directly off the loaded modelscope DfsmnAns module: linear/project
  bias+weight and the depthwise conv1 kernel), avoiding modelscope's own
  "recompute the whole utterance from scratch on a sliding window" approach
  in favor of true O(1)-per-hop incremental streaming.
- STFT/ISTFT: n_fft=hop*2=1920, hamming window (periodic=False), matching
  the upstream pipeline's torch.stft(..., center=False) / librosa.istft(...,
  center=False) pair. Reimplemented via cos/sin DFT matmul (same style as
  the fbank DFT and this project's DPDFNet export) plus an explicit
  synthesis normalization envelope (w(p)^2 + w(p+hop)^2 for p in
  [0, hop)), since a plain hamming window does not satisfy the
  squared-COLA=1 identity that a window like DPDFNet's Vorbis window does;
  librosa.istft normalizes by exactly this per-sample envelope internally.

Streaming
---------
- streaming=1 in ONNX metadata; state_size/hop_size are also stored so the
  C++ backend can size its buffers without hardcoding them.
- Natural minimal chunk is hop_size=960 samples (20 ms). Whole-utterance
  (offline) decoding is just repeated hop-sized calls starting from
  zeroed caches.
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnxruntime.quantization import QuantType, quantize_dynamic

OPSET_VERSION = 18

SAMPLE_RATE = 48000
HOP_SIZE = 960
N_FFT = 1920
FBANK_FRAME_LEN = 1920
FBANK_PADDED_LEN = 2048
NUM_MEL_BINS = 120
NUM_FBANK_FREQ_BINS = FBANK_PADDED_LEN // 2 + 1
NUM_STFT_FREQ_BINS = N_FFT // 2 + 1
FBANK_SCALE = 32768.0
PREEMPHASIS_COEFF = 0.97
LOG_EPSILON = float(np.finfo(np.float32).eps)


def load_dfsmn_ans_psm_48k(model_dir):
    import json

    from modelscope.models.audio.ans.denoise_net import DfsmnAns

    with open(os.path.join(model_dir, "configuration.json")) as f:
        cfg = json.load(f)["model"]
    cfg = dict(cfg)
    cfg.pop("type", None)

    model = DfsmnAns(model_dir=model_dir, **cfg)
    state_dict = torch.load(
        os.path.join(model_dir, "pytorch_model.bin"), map_location="cpu", weights_only=True
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"load_state_dict missing={missing} unexpected={unexpected}")
    return model.eval()


def dft_matrices(n_fft: int, num_freq_bins: int):
    samples = torch.arange(n_fft, dtype=torch.float32)
    freqs = torch.arange(num_freq_bins, dtype=torch.float32)
    angles = 2.0 * math.pi * freqs[:, None] * samples[None, :] / n_fft
    real = torch.cos(angles)
    imag = -torch.sin(angles)
    return real, imag


def inverse_scale(n_fft: int, num_freq_bins: int):
    scale = torch.full((num_freq_bins,), 2.0 / n_fft)
    scale[0] = 1.0 / n_fft
    if n_fft % 2 == 0:
        scale[-1] = 1.0 / n_fft
    return scale


class StreamingDfsmnAnsPsm48k(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.lorder = int(model.lorder)
        self.fsmn_depth = len(model.deepfsmn)
        self.hidden_size = int(model.deepfsmn[0].hidden_size)
        self.hop_size = HOP_SIZE
        self.n_fft = N_FFT

        # --- DFSMN mask network weights, pulled directly off the loaded
        # modelscope module so we can hand-roll a stateful per-frame forward.
        self.linear1_weight = nn.Parameter(model.linear1.linear.weight.detach().clone(), requires_grad=False)
        self.linear1_bias = nn.Parameter(model.linear1.linear.bias.detach().clone(), requires_grad=False)
        self.linear2_weight = nn.Parameter(model.linear2.linear.weight.detach().clone(), requires_grad=False)
        self.linear2_bias = nn.Parameter(model.linear2.linear.bias.detach().clone(), requires_grad=False)

        fsmn_linear_weight = []
        fsmn_linear_bias = []
        fsmn_project_weight = []
        fsmn_conv_weight = []
        for layer in model.deepfsmn:
            fsmn_linear_weight.append(layer.linear.weight.detach().clone())
            fsmn_linear_bias.append(layer.linear.bias.detach().clone())
            fsmn_project_weight.append(layer.project.weight.detach().clone())
            # conv1.weight: [channels, 1, lorder, 1] -> [channels, lorder]
            fsmn_conv_weight.append(layer.conv1.weight.detach().clone().squeeze(-1).squeeze(1))
        self.fsmn_linear_weight = nn.Parameter(torch.stack(fsmn_linear_weight), requires_grad=False)
        self.fsmn_linear_bias = nn.Parameter(torch.stack(fsmn_linear_bias), requires_grad=False)
        self.fsmn_project_weight = nn.Parameter(torch.stack(fsmn_project_weight), requires_grad=False)
        self.fsmn_conv_weight = nn.Parameter(torch.stack(fsmn_conv_weight), requires_grad=False)

        # --- fbank frontend buffers.
        fbank_window = torch.hamming_window(FBANK_FRAME_LEN, periodic=False, alpha=0.54, beta=0.46)
        fbank_real, fbank_imag = dft_matrices(FBANK_PADDED_LEN, NUM_FBANK_FREQ_BINS)

        from torchaudio.compliance.kaldi import get_mel_banks

        mel_matrix, _ = get_mel_banks(
            NUM_MEL_BINS, FBANK_PADDED_LEN, float(SAMPLE_RATE), 20.0, 0.0, 100.0, -500.0, 1.0
        )
        mel_matrix = F.pad(mel_matrix, (0, 1), mode="constant", value=0.0)  # [120, 1025]

        self.register_buffer("fbank_window", fbank_window)
        self.register_buffer("fbank_dft_real", fbank_real)
        self.register_buffer("fbank_dft_imag", fbank_imag)
        self.register_buffer("mel_matrix_t", mel_matrix.transpose(0, 1).contiguous())  # [1025, 120]

        # --- STFT/ISTFT buffers (n_fft=1920, hop=960, hamming window).
        stft_window = torch.hamming_window(N_FFT, periodic=False, alpha=0.54, beta=0.46)
        stft_real, stft_imag = dft_matrices(N_FFT, NUM_STFT_FREQ_BINS)
        syn_scale = inverse_scale(N_FFT, NUM_STFT_FREQ_BINS)

        self.register_buffer("stft_window", stft_window)
        self.register_buffer("stft_analysis_real", stft_real)
        self.register_buffer("stft_analysis_imag", stft_imag)
        self.register_buffer("stft_synthesis_real", (stft_real * syn_scale[:, None]).contiguous())
        self.register_buffer("stft_synthesis_imag", (stft_imag * syn_scale[:, None]).contiguous())

        synthesis_norm = stft_window[:HOP_SIZE] ** 2 + stft_window[HOP_SIZE:] ** 2
        self.register_buffer("synthesis_norm", synthesis_norm)

    def state_size(self):
        return self.fsmn_depth * (self.lorder - 1) * self.hidden_size

    def compute_fbank(self, frame: torch.Tensor) -> torch.Tensor:
        # frame: [1, FBANK_FRAME_LEN], already scaled to int16 range.
        x = frame - frame.mean(dim=1, keepdim=True)
        prev = F.pad(x.unsqueeze(0), (1, 0), mode="replicate").squeeze(0)[:, :-1]
        x = x - PREEMPHASIS_COEFF * prev
        x = x * self.fbank_window.unsqueeze(0)
        x = F.pad(x, (0, FBANK_PADDED_LEN - FBANK_FRAME_LEN), mode="constant", value=0.0)
        real = torch.matmul(x, self.fbank_dft_real.transpose(0, 1))
        imag = torch.matmul(x, self.fbank_dft_imag.transpose(0, 1))
        power = real * real + imag * imag
        mel = torch.matmul(power, self.mel_matrix_t)
        mel = torch.clamp(mel, min=LOG_EPSILON)
        return torch.log(mel)

    def dfsmn_step(self, fbank: torch.Tensor, cache_in: torch.Tensor):
        # fbank: [1, 120]. cache_in: [fsmn_depth, lorder - 1, hidden_size].
        x1 = torch.matmul(fbank, self.linear1_weight.transpose(0, 1)) + self.linear1_bias
        x = F.relu(x1)

        cache_out_layers = []
        for i in range(self.fsmn_depth):
            f1 = F.relu(torch.matmul(x, self.fsmn_linear_weight[i].transpose(0, 1)) + self.fsmn_linear_bias[i])
            p1 = torch.matmul(f1, self.fsmn_project_weight[i].transpose(0, 1))  # [1, hidden]
            window = torch.cat([cache_in[i], p1], dim=0)  # [lorder, hidden]
            conv_out = torch.sum(window * self.fsmn_conv_weight[i].transpose(0, 1), dim=0, keepdim=True)
            out = p1 + conv_out
            x = x + out
            cache_out_layers.append(window[1:])

        mask = torch.matmul(x, self.linear2_weight.transpose(0, 1)) + self.linear2_bias
        mask = torch.sigmoid(mask)
        cache_out = torch.stack(cache_out_layers, dim=0)
        return mask, cache_out

    def forward(
        self,
        speech: torch.Tensor,
        analysis_cache: torch.Tensor,
        synthesis_cache: torch.Tensor,
        state_in: torch.Tensor,
    ):
        frame = torch.cat([analysis_cache, speech], dim=1)  # [1, 1920]

        fbank = self.compute_fbank(frame * FBANK_SCALE)  # [1, 120]
        fsmn_cache_in = state_in.view(self.fsmn_depth, self.lorder - 1, self.hidden_size)
        mask, fsmn_cache_out = self.dfsmn_step(fbank, fsmn_cache_in)  # mask: [1, 961]
        state_out = fsmn_cache_out.reshape(-1)

        windowed = frame * self.stft_window.unsqueeze(0)
        spec_real = torch.matmul(windowed, self.stft_analysis_real.transpose(0, 1))
        spec_imag = torch.matmul(windowed, self.stft_analysis_imag.transpose(0, 1))
        spec_real = spec_real * mask
        spec_imag = spec_imag * mask

        enhanced_frame = torch.matmul(spec_real, self.stft_synthesis_real) + torch.matmul(
            spec_imag, self.stft_synthesis_imag
        )
        enhanced_frame = enhanced_frame * self.stft_window.unsqueeze(0)

        enhanced = (enhanced_frame[:, : self.hop_size] + synthesis_cache) / self.synthesis_norm.unsqueeze(0)
        synthesis_cache_out = enhanced_frame[:, self.hop_size :]
        analysis_cache_out = speech

        return enhanced, analysis_cache_out, synthesis_cache_out, state_out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export streaming DFSMN-ANS-PSM 48k causal denoise model to ONNX."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/data/user/lxp/llm/downloads/models/iic/speech_dfsmn_ans_psm_48k_causal",
        help="Path to the modelscope checkpoint directory.",
    )
    parser.add_argument("--output", type=str, default="public/models/dfsmn_ans_psm_48k.onnx")
    parser.add_argument("--opset", type=int, default=OPSET_VERSION)
    parser.add_argument("--verify-frames", type=int, default=40)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument(
        "--quantize", type=int, default=1, help="Save dynamic int8 model next to ONNX using .onnx -> .int8.onnx."
    )
    return parser.parse_args()


def initial_inputs(model: StreamingDfsmnAnsPsm48k):
    return (
        torch.zeros(1, model.hop_size, dtype=torch.float32),
        torch.zeros(1, model.hop_size, dtype=torch.float32),
        torch.zeros(1, model.hop_size, dtype=torch.float32),
        torch.zeros(model.state_size(), dtype=torch.float32),
    )


def add_metadata(output_path: str, state_size: int):
    model = onnx.load(output_path)
    metadata = {
        "model_type": "dfsmn_ans_psm_48k_denoise",
        "sample_rate": str(SAMPLE_RATE),
        "frame_length": str(N_FFT),
        "frame_shift": str(HOP_SIZE),
        "window_type": "hamming",
        "state_size": str(state_size),
        "delay_hops": "1",
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


def reference_forward(model: nn.Module, waveform: torch.Tensor) -> torch.Tensor:
    """Ground-truth offline reference: upstream pipeline's own ops
    (torchaudio kaldi fbank, torch.stft, librosa.istft), dither disabled."""
    import librosa
    import torchaudio

    with torch.no_grad():
        audio_in = waveform.unsqueeze(0) * FBANK_SCALE
        fbanks = torchaudio.compliance.kaldi.fbank(
            audio_in,
            dither=0.0,
            frame_length=40.0,
            frame_shift=20.0,
            num_mel_bins=NUM_MEL_BINS,
            sample_frequency=SAMPLE_RATE,
            window_type="hamming",
        )
        fbanks = fbanks.unsqueeze(0)
        masks = model(fbanks)

        window = torch.hamming_window(N_FFT, periodic=False)
        spectrum = torch.stft(waveform, N_FFT, HOP_SIZE, N_FFT, center=False, window=window, return_complex=False)
        masks = masks.permute(2, 1, 0)
        masked_spec = spectrum * masks

    masked_spec = masked_spec.numpy()
    masked_spec_complex = masked_spec[:, :, 0] + 1j * masked_spec[:, :, 1]
    enhanced = librosa.istft(
        masked_spec_complex,
        hop_length=HOP_SIZE,
        win_length=N_FFT,
        window="hamming",
        center=False,
        length=len(waveform),
    )
    return torch.from_numpy(enhanced)


def verify(source_model: nn.Module, streaming_model: StreamingDfsmnAnsPsm48k, output_path: str, num_frames: int):
    torch.manual_seed(20260815)
    hop_size = streaming_model.hop_size
    # The streaming path has an inherent one-hop algorithmic latency: the
    # hop emitted by call i covers absolute output samples
    # [(i-1)*hop, i*hop) (it's the newly-finalized overlap-add region of the
    # analysis frame built from the *previous* hop's history + this hop's
    # new samples), whereas the offline reference has no such per-call
    # framing. Skip the first hop (no valid output yet, zeroed history) and
    # shift the comparison index back by one hop accordingly.
    warmup_hops = 1
    latency_hops = 1
    waveform = torch.rand(1, num_frames * hop_size, dtype=torch.float32) * 2.0 - 1.0

    reference = reference_forward(source_model, waveform.squeeze(0))

    torch_state = list(initial_inputs(streaming_model)[1:])
    ort_state = [value.numpy().copy() for value in initial_inputs(streaming_model)[1:]]
    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])

    waveform_diff = 0.0
    ref_diff = 0.0
    state_diff = 0.0
    for hop_index, offset in enumerate(range(0, waveform.shape[1], hop_size)):
        speech = waveform[:, offset : offset + hop_size]
        with torch.no_grad():
            torch_outputs = streaming_model(speech, *torch_state)
        feeds = {
            "speech": speech.numpy(),
            "analysis_cache": ort_state[0],
            "synthesis_cache": ort_state[1],
            "state_in": ort_state[2],
        }
        ort_outputs = session.run(None, feeds)
        waveform_diff = max(waveform_diff, float(np.max(np.abs(torch_outputs[0].numpy() - ort_outputs[0]))))
        for torch_cache, ort_cache in zip(torch_outputs[1:], ort_outputs[1:]):
            state_diff = max(state_diff, float(np.max(np.abs(torch_cache.numpy() - ort_cache))))

        if hop_index >= warmup_hops:
            ref_offset = offset - latency_hops * hop_size
            ref_chunk = reference[ref_offset : ref_offset + hop_size]
            ref_diff = max(ref_diff, float(np.max(np.abs(torch_outputs[0].numpy()[0] - ref_chunk.numpy()))))

        torch_state = [value.detach().clone() for value in torch_outputs[1:]]
        ort_state = [value.copy() for value in ort_outputs[1:]]

    print(
        f"PyTorch vs ONNX max abs diff: waveform={waveform_diff:.8g}, state={state_diff:.8g}; "
        f"streaming vs offline-reference max abs diff (after warm-up)={ref_diff:.8g}"
    )
    if waveform_diff >= 1e-4 or state_diff >= 1e-3:
        raise RuntimeError("PyTorch/ONNX maximum absolute difference exceeds threshold")
    if ref_diff >= 1e-2:
        raise RuntimeError("Streaming vs offline-reference maximum absolute difference exceeds 1e-2")


def quantize_onnx_model(input_path: str, output_path: str):
    model = onnx.load(input_path)
    nodes_to_exclude = []
    preprocess_keywords = (
        "fbank_window",
        "fbank_dft",
        "mel_matrix",
        "stft_window",
        "stft_analysis",
        "stft_synthesis",
        "synthesis_norm",
    )
    preprocess_inits = [
        init.name for init in model.graph.initializer if any(k in init.name.lower() for k in preprocess_keywords)
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

    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,
        nodes_to_exclude=nodes_to_exclude,
        per_channel=False,
        reduce_range=False,
    )
    print(f"Quantized int8 model saved to: {output_path}")


def export(streaming_model: StreamingDfsmnAnsPsm48k, output_path: str, quantize: bool):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy_inputs = initial_inputs(streaming_model)
    torch.onnx.export(
        streaming_model,
        dummy_inputs,
        str(output),
        input_names=["speech", "analysis_cache", "synthesis_cache", "state_in"],
        output_names=["enhanced", "analysis_cache_out", "synthesis_cache_out", "state_out"],
        opset_version=OPSET_VERSION,
        dynamo=False,
    )
    state_size = streaming_model.state_size()
    add_metadata(str(output), state_size)
    print(f"Exported DFSMN-ANS-PSM 48k waveform model to: {output}")
    size = os.path.getsize(output)
    print(f"File size: {size:,} bytes ({size / 1024 / 1024:.2f} MB)")

    if quantize:
        quantized_path = str(output).replace(".onnx", ".int8.onnx")
        if quantized_path == str(output):
            raise ValueError(f"ONNX path must end with .onnx for int8 output: {output}")
        quantize_onnx_model(str(output), quantized_path)
        add_metadata(quantized_path, state_size)
        size = os.path.getsize(quantized_path)
        print(f"Int8 file size: {size:,} bytes ({size / 1024 / 1024:.2f} MB)")


def main():
    args = parse_args()
    if args.verify_frames <= 0:
        raise ValueError("--verify-frames must be greater than zero")
    source_model = load_dfsmn_ans_psm_48k(args.model_dir)
    streaming_model = StreamingDfsmnAnsPsm48k(source_model).eval()
    export(streaming_model, args.output, quantize=bool(args.quantize))
    if not args.no_verify:
        verify(source_model, streaming_model, args.output, args.verify_frames)


if __name__ == "__main__":
    main()
