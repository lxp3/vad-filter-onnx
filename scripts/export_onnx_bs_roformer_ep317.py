#!/usr/bin/env python3
"""Export BS-Roformer ep317 as a fixed 8-second waveform ONNX graph.

The graph contains one non-streaming chunk: reflect-padded Hann STFT, the
band-split Roformer, complex mask multiplication, and overlap-add iSTFT.
Hamming overlap across chunks stays outside the graph. Audio shorter than one
chunk is zero-padded on the right by the C++ runner, which is not the same as
python-audio-separator's variable-length forward.
"""

import argparse
import math
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F
import yaml
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SEPARATOR_ROOT = REPO_ROOT / "debug" / "python-audio-separator"
sys.path.insert(0, str(SEPARATOR_ROOT))

from audio_separator.separator.uvr_lib_v5.roformer.attend import Attend
from audio_separator.separator.uvr_lib_v5.roformer.bs_roformer import BSRoformer
import audio_separator.separator.uvr_lib_v5.roformer.bs_roformer as bs_roformer_module

YAML_URL = (
    "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/"
    "main/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml"
)
SAMPLE_RATE = 44100
CHANNELS = 2
STEMS = 1
N_FFT = 2048
HOP = 441
FRAMES = 801
CHUNK = HOP * (FRAMES - 1)
OVERLAP = 4
STEP = CHUNK // OVERLAP
PEAK = np.float32(0.9)
WEIGHT_FLOOR = np.float32(1e-10)


class FixedRealStft(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_fft = N_FFT
        self.hop = HOP
        self.frames = FRAMES
        self.pad = N_FFT // 2
        self.chunk = CHUNK
        self.out_len = (FRAMES - 1) * HOP + N_FFT
        window = torch.hann_window(N_FFT, periodic=True)
        basis = torch.fft.rfft(torch.eye(N_FFT, dtype=window.dtype) * window, dim=-1)
        self.register_buffer("analysis_real", basis.real.transpose(0, 1).unsqueeze(1).contiguous())
        self.register_buffer("analysis_imag", basis.imag.transpose(0, 1).unsqueeze(1).contiguous())
        sample_index = torch.arange(N_FFT, dtype=torch.float64)
        bin_index = torch.arange(N_FFT // 2 + 1, dtype=torch.float64)
        angles = 2 * math.pi * bin_index[:, None] * sample_index[None, :] / N_FFT
        scale = torch.ones(N_FFT // 2 + 1, 1, dtype=torch.float64)
        scale[1:-1] = 2
        self.register_buffer("inverse_real", (torch.cos(angles) * scale / N_FFT).float())
        self.register_buffer("inverse_imag", (-torch.sin(angles) * scale / N_FFT).float())
        self.register_buffer("window", window)

    def stft(self, wave):
        padded = F.pad(wave.unsqueeze(1), (self.pad, self.pad), mode="reflect")
        real = F.conv1d(padded, self.analysis_real, stride=self.hop)
        imag = F.conv1d(padded, self.analysis_imag, stride=self.hop)
        return torch.stack((real, imag), dim=-1)

    def istft(self, spec):
        real = spec[..., 0]
        imag = spec[..., 1]
        frames = torch.matmul(real.transpose(1, 2), self.inverse_real)
        frames = frames + torch.matmul(imag.transpose(1, 2), self.inverse_imag)
        frames = frames * self.window.view(1, 1, -1)
        frames = frames.transpose(1, 2).contiguous()
        window_sq = self.window.square().view(1, self.n_fft, 1).repeat(CHANNELS, 1, self.frames)
        folded = F.fold(
            frames,
            output_size=(1, self.out_len),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop),
        )
        envelope = F.fold(
            window_sq,
            output_size=(1, self.out_len),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop),
        )
        restored = (folded / envelope.clamp(min=1e-11)).reshape(CHANNELS, self.out_len)
        return restored[:, self.pad : self.pad + self.chunk]


class Ep317OnnxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        if model.num_stems != STEMS or model.audio_channels != CHANNELS:
            raise RuntimeError("ep317 wrapper only supports one stereo stem")
        self.model = model
        self.frontend = FixedRealStft()
        self.bands = len(model.band_split.dim_inputs)
        self.dim = int(model.final_norm.gamma.numel())
        self.freq_bins = N_FFT // 2 + 1
        self.merged_bins = self.freq_bins * CHANNELS

    def forward(self, waveform):
        spec = self.frontend.stft(waveform.reshape(CHANNELS, CHUNK))
        spec = spec.reshape(1, CHANNELS, self.freq_bins, FRAMES, 2)
        spec = spec.permute(0, 2, 1, 3, 4).reshape(1, self.merged_bins, FRAMES, 2)
        features = spec.permute(0, 2, 1, 3).reshape(1, FRAMES, self.merged_bins * 2)
        hidden = self.model.band_split(features)
        for transformer_block in self.model.layers:
            time_transformer, freq_transformer = transformer_block
            hidden = hidden.permute(0, 2, 1, 3).reshape(self.bands, FRAMES, self.dim)
            hidden = time_transformer(hidden)
            hidden = hidden.reshape(1, self.bands, FRAMES, self.dim).permute(0, 2, 1, 3)
            hidden = hidden.reshape(FRAMES, self.bands, self.dim)
            hidden = freq_transformer(hidden)
            hidden = hidden.reshape(1, FRAMES, self.bands, self.dim)
        hidden = self.model.final_norm(hidden)
        masks = [estimator(hidden) for estimator in self.model.mask_estimators]
        mask = torch.stack(masks, dim=1)
        mask = mask.reshape(1, STEMS, FRAMES, self.merged_bins, 2).permute(0, 1, 3, 2, 4)
        mixture = spec.unsqueeze(1)
        real = mixture[..., 0] * mask[..., 0] - mixture[..., 1] * mask[..., 1]
        imag = mixture[..., 0] * mask[..., 1] + mixture[..., 1] * mask[..., 0]
        real = real.reshape(1, STEMS, self.freq_bins, CHANNELS, FRAMES)
        imag = imag.reshape(1, STEMS, self.freq_bins, CHANNELS, FRAMES)
        real = real.permute(0, 1, 3, 2, 4).reshape(CHANNELS, self.freq_bins, FRAMES)
        imag = imag.permute(0, 1, 3, 2, 4).reshape(CHANNELS, self.freq_bins, FRAMES)
        separated = self.frontend.istft(torch.stack((real, imag), dim=-1))
        return separated.reshape(1, STEMS, CHANNELS, CHUNK)[:, 0]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(SEPARATOR_ROOT / "models" / "model_bs_roformer_ep_317_sdr_12.9755.ckpt"),
    )
    parser.add_argument("--output", default=str(REPO_ROOT / "public" / "models" / "bs_roformer_ep317.onnx"))
    parser.add_argument("--cpp-bin", default="")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def ensure_yaml(checkpoint):
    yaml_path = Path(checkpoint).with_suffix(".yaml")
    if not yaml_path.exists():
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading config to {yaml_path}", flush=True)
        urllib.request.urlretrieve(YAML_URL, yaml_path)
    return yaml_path


def load_model(checkpoint):
    config = yaml.load(ensure_yaml(checkpoint).read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    model_config = dict(config["model"])
    model_config["flash_attn"] = False
    if "freqs_per_bands" in model_config:
        model_config["freqs_per_bands"] = tuple(model_config["freqs_per_bands"])
    if "multi_stft_resolutions_window_sizes" in model_config:
        model_config["multi_stft_resolutions_window_sizes"] = tuple(
            model_config["multi_stft_resolutions_window_sizes"]
        )
    if sum(model_config["freqs_per_bands"]) != N_FFT // 2 + 1:
        raise RuntimeError("frequency bands do not match the 2048-point STFT")
    model = BSRoformer(**model_config)
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    for module in model.modules():
        if isinstance(module, Attend):
            module.flash = False
        cached = getattr(module, "cached_freqs", None)
        if cached is not None and float(cached.abs().max()) != 0.0:
            raise RuntimeError("rotary cache is not the identity used by this BSRoformer build")
    # This tree's rotary helper returns the zero-initialized cache for every
    # sequence up to 8192, so both time and frequency rotations are identities.
    # Keep that behavior and avoid apply_rotary_emb's empty concatenations.
    bs_roformer_module._rotate_queries_or_keys = lambda rotary_embed, tensor: tensor
    return model


def abs_stats(left, right):
    difference = np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))
    return float(difference.max()), float(difference.mean())


def report(name, left, right):
    maximum, mean = abs_stats(left, right)
    print(f"METRIC {name} max={maximum:.8g} mean={mean:.8g}", flush=True)
    return maximum, mean


def check_frontend():
    torch.manual_seed(20261016)
    wave = torch.randn(CHANNELS, CHUNK)
    window = torch.hann_window(N_FFT, periodic=True)
    reference = torch.stft(
        wave,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        window=window,
        center=True,
        return_complex=True,
        normalized=False,
    )
    frontend = FixedRealStft()
    spec = frontend.stft(wave)
    maximum, mean = abs_stats(spec[..., 0].detach(), reference.real)
    imag_max, imag_mean = abs_stats(spec[..., 1].detach(), reference.imag)
    restored = torch.istft(
        reference, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, window=window, center=True, length=CHUNK
    )
    wave_max, wave_mean = abs_stats(frontend.istft(spec).detach(), restored)
    print(
        f"METRIC stft_real max={maximum:.8g} mean={mean:.8g}",
        flush=True,
    )
    print(f"METRIC stft_imag max={imag_max:.8g} mean={imag_mean:.8g}", flush=True)
    print(f"METRIC istft max={wave_max:.8g} mean={wave_mean:.8g}", flush=True)
    if max(maximum, imag_max) > 1e-3 or wave_max > 1e-4:
        raise RuntimeError("real DFT frontend does not match torch.stft/istft")


def symmetric_hamming(size):
    if size == 1:
        return np.ones(1, dtype=np.float32)
    index = np.arange(size, dtype=np.float64)
    window = 0.54 - 0.46 * np.cos(2 * np.pi * index / (size - 1))
    return window.astype(np.float32)


def chunk_starts(audio_length):
    if audio_length <= CHUNK:
        return [0]
    starts = []
    offset = 0
    while offset < audio_length:
        if offset + CHUNK >= audio_length:
            tail = max(audio_length - CHUNK, 0)
            if not starts or starts[-1] != tail:
                starts.append(tail)
            break
        starts.append(offset)
        offset += STEP
    return starts


def limit_peak(wave):
    peak = np.max(np.abs(wave))
    if peak > PEAK:
        wave = wave * (PEAK / np.float32(peak))
    return wave.astype(np.float32, copy=False)


def overlap_add(session, mix):
    length = mix.shape[1]
    vocals = np.zeros((CHANNELS, length), dtype=np.float32)
    weight = np.zeros(length, dtype=np.float32)
    window = symmetric_hamming(CHUNK)
    input_name = session.get_inputs()[0].name
    for start in chunk_starts(length):
        valid = min(CHUNK, length - start)
        chunk = np.zeros((1, CHANNELS, CHUNK), dtype=np.float32)
        chunk[0, :, :valid] = mix[:, start : start + valid]
        output = session.run(None, {input_name: chunk})[0][0]
        vocals[:, start : start + valid] += output[:, :valid] * window[:valid]
        weight[start : start + valid] += window[:valid]
    vocals /= np.maximum(weight, WEIGHT_FLOOR)
    instrumental = mix - vocals
    return limit_peak(vocals), limit_peak(instrumental)


def add_metadata(output_path):
    model = onnx.load(output_path)
    metadata = {
        "model_type": "bs_roformer_separate",
        "sample_rate": str(SAMPLE_RATE),
        "channels": str(CHANNELS),
        "chunk_size": str(CHUNK),
        "overlap": str(OVERLAP),
        "streaming": "0",
    }
    del model.metadata_props[:]
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.checker.check_model(model)
    onnx.save(model, output_path)


def export_graph(wrapper, output_path):
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, CHANNELS, CHUNK, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy,
        str(destination),
        input_names=["waveform"],
        output_names=["vocals"],
        opset_version=18,
        dynamo=False,
        do_constant_folding=True,
    )
    add_metadata(str(destination))
    print(f"Exported {destination} ({destination.stat().st_size / 1024 / 1024:.1f} MB)", flush=True)


def ort_session(output_path, threads):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_cpu_mem_arena = False
    return ort.InferenceSession(output_path, sess_options=options, providers=["CPUExecutionProvider"])


def compare_overlap(session, cpp_bin, threads):
    generator = np.random.default_rng(20261016)
    frames = CHUNK + STEP
    mix = generator.standard_normal((frames, CHANNELS)).astype(np.float32) * np.float32(0.2)
    mix[0, 0] = np.float32(1.25)
    packed = np.ascontiguousarray(mix.T)
    vocals, instrumental = overlap_add(session, limit_peak(packed.copy()))
    with tempfile.TemporaryDirectory(prefix="bs-roformer-") as directory:
        root = Path(directory)
        raw_path = root / "mix.f32"
        vocals_path = root / "vocals.f32"
        instrumental_path = root / "instrumental.f32"
        mix.astype(np.float32).tofile(raw_path)
        subprocess.check_call(
            [
                cpp_bin,
                "--model-path",
                session._model_path,
                "--input-raw",
                str(raw_path),
                "--frames",
                str(frames),
                "--channels",
                str(CHANNELS),
                "--sample-rate",
                str(SAMPLE_RATE),
                "--output-vocals",
                str(vocals_path),
                "--output-instrumental",
                str(instrumental_path),
                "--num-threads",
                str(threads),
            ]
        )
        cpp_vocals = np.fromfile(vocals_path, dtype=np.float32).reshape(frames, CHANNELS)
        cpp_instrumental = np.fromfile(instrumental_path, dtype=np.float32).reshape(frames, CHANNELS)
    report("ola_vocals", vocals.T, cpp_vocals)
    report("ola_instrumental", instrumental.T, cpp_instrumental)


def main():
    args = parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    check_frontend()
    print("Loading checkpoint", flush=True)
    model = load_model(args.checkpoint)
    wrapper = Ep317OnnxWrapper(model).eval()
    torch.manual_seed(20261016)
    waveform = torch.randn(1, CHANNELS, CHUNK)
    print("Comparing PyTorch forward and real-DFT wrapper", flush=True)
    with torch.no_grad():
        reference = model(waveform)
        wrapped = wrapper(waveform)
    torch_max, _ = report("torch_vs_real", reference, wrapped)
    if torch_max > 1e-2:
        raise RuntimeError("real-DFT wrapper diverges from BSRoformer.forward")
    if not args.skip_export or not Path(args.output).exists():
        print("Exporting ONNX", flush=True)
        export_graph(wrapper, args.output)
    elif args.skip_export:
        print(f"Reusing {args.output}", flush=True)
    session = ort_session(args.output, args.threads)
    session._model_path = args.output
    print("Comparing real-DFT wrapper and ONNX Runtime", flush=True)
    ort_vocals = session.run(None, {session.get_inputs()[0].name: waveform.numpy()})[0]
    ort_max, _ = report("real_vs_ort", wrapped.detach(), ort_vocals)
    if ort_max > 1e-2:
        raise RuntimeError("ONNX Runtime diverges from the real-DFT wrapper")
    if args.cpp_bin:
        print("Comparing Python overlap-add and C++", flush=True)
        compare_overlap(session, args.cpp_bin, args.threads)


if __name__ == "__main__":
    main()
