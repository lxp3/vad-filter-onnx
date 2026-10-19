#!/usr/bin/env python3
"""Export a streaming ONNX graph for hush_dfnet_16k.

Feature extraction (Vorbis STFT, ERB, exponential normalization) is inside the
graph. Each call consumes one hop of float32 waveform plus streaming caches.
"""

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import avioflow
from onnxruntime.quantization import QuantType, quantize_dynamic

OPSET_VERSION = 18
REPO_ROOT = Path(__file__).resolve().parents[1]
HUSH_ROOT = REPO_ROOT / "debug" / "Hush"
DEFAULT_CHECKPOINT = HUSH_ROOT / "deployment" / "models" / "model_best.ckpt"
DEFAULT_ONNX = REPO_ROOT / "public" / "models" / "hush_dfnet_16k.onnx"
MODEL_TYPE = "hush_dfnet_16k"


def _load_hush_symbols():
    if str(HUSH_ROOT) not in sys.path:
        sys.path.insert(0, str(HUSH_ROOT))
    from model.dfnet_se import DfNetSE, get_config, get_norm_alpha  # noqa: WPS433

    infer_path = HUSH_ROOT / "scripts" / "infer_single.py"
    spec = importlib.util.spec_from_file_location("hush_infer_single", infer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return DfNetSE, get_config, get_norm_alpha, module.enhance


def vorbis_window(window_len: int) -> torch.Tensor:
    half = window_len / 2
    indices = torch.arange(window_len, dtype=torch.float32)
    s = torch.sin(0.5 * math.pi * (indices + 0.5) / half)
    return torch.sin(0.5 * math.pi * s * s)


class StreamingSTFT(nn.Module):
    def __init__(self, n_fft: int, hop_size: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        freq_bins = n_fft // 2 + 1

        samples = torch.arange(n_fft, dtype=torch.float32)
        frequencies = torch.arange(freq_bins, dtype=torch.float32)
        angles = 2.0 * math.pi * frequencies[:, None] * samples[None, :] / n_fft
        window = vorbis_window(n_fft)

        inverse_scale = torch.full((freq_bins,), 2.0 / n_fft)
        inverse_scale[0] = 1.0 / n_fft
        inverse_scale[-1] = 1.0 / n_fft
        wnorm = 1.0 / (n_fft**2 / (2 * hop_size))

        self.register_buffer("stft_analysis_real", torch.cos(angles))
        self.register_buffer("stft_analysis_imag", -torch.sin(angles))
        self.register_buffer("stft_synthesis_real", torch.cos(angles) * inverse_scale[:, None])
        self.register_buffer("stft_synthesis_imag", -torch.sin(angles) * inverse_scale[:, None])
        self.register_buffer("stft_window", window)
        self.register_buffer("stft_wnorm", torch.tensor(wnorm, dtype=torch.float32))
        self.register_buffer("stft_inv_wnorm", torch.tensor(1.0 / wnorm, dtype=torch.float32))

    def analysis(self, speech, analysis_cache):
        frame = torch.cat([analysis_cache, speech], dim=1)
        windowed = frame * self.stft_window
        real = torch.matmul(windowed, self.stft_analysis_real.transpose(0, 1))
        imag = torch.matmul(windowed, self.stft_analysis_imag.transpose(0, 1))
        spec = torch.stack([real, imag], dim=-1) * self.stft_wnorm
        spec = spec.unsqueeze(1).unsqueeze(1)
        return spec, speech

    def synthesis(self, spec_e, synthesis_cache):
        spec_e = spec_e.squeeze(1).squeeze(1) * self.stft_inv_wnorm
        enhanced_frame = (
            torch.matmul(spec_e[..., 0], self.stft_synthesis_real)
            + torch.matmul(spec_e[..., 1], self.stft_synthesis_imag)
        ) * self.stft_window
        hop = self.hop_size
        enhanced = enhanced_frame[:, :hop] + synthesis_cache
        synthesis_cache_out = enhanced_frame[:, hop:]
        return enhanced, synthesis_cache_out


class CausalConv(nn.Module):
    def __init__(self, seq: nn.Sequential, cache_shape):
        super().__init__()
        children = list(seq.children())
        if len(children) > 0 and isinstance(children[0], nn.ConstantPad2d):
            pad = children[0].padding
            self.hist = int(pad[2] + pad[3])
            self.rest = nn.Sequential(*children[1:])
        else:
            self.hist = 0
            self.rest = nn.Sequential(*children)
        self.cache_shape = tuple(cache_shape)

    def forward(self, x, cache):
        if self.hist == 0:
            return self.rest(x), cache
        buf = torch.cat([cache, x], dim=2)
        y = self.rest(buf)
        new_cache = buf[:, :, 1:, :]
        return y, new_cache


def capture_input_shapes(root, names, runner):
    shapes = {}
    named = dict(root.named_modules())
    hooks = []

    def make_hook(name):
        def hook(_module, inp):
            shapes[name] = tuple(inp[0].shape)

        return hook

    for name in names:
        hooks.append(named[name].register_forward_pre_hook(make_hook(name)))
    with torch.no_grad():
        runner()
    for hook in hooks:
        hook.remove()
    return shapes


def wrap_causal_layers(root, names, runner):
    shapes = capture_input_shapes(root, names, runner)
    named = dict(root.named_modules())
    wrapped = {}
    for name in names:
        seq = named[name]
        _batch, channels, _time, freq = shapes[name]
        children = list(seq.children())
        hist = 0
        if len(children) > 0 and isinstance(children[0], nn.ConstantPad2d):
            pad = children[0].padding
            hist = int(pad[2] + pad[3])
        wrapped[name] = CausalConv(seq, (1, channels, hist, freq))
    return wrapped


class FeatureState(nn.Module):
    """libdf erb()/erb_norm()/unit_norm(), with the non-zero init folded in.

    External norm state is a delta from libdf's initial ramp, so an all-zero
    state vector matches a fresh libdf session.
    """

    def __init__(self, erb_fb_mat: torch.Tensor, nb_erb: int, nb_df: int, alpha: float):
        super().__init__()
        self.register_buffer("erb_fb", erb_fb_mat)
        self.register_buffer("erb_init", torch.linspace(-60.0, -90.0, nb_erb))
        self.register_buffer("spec_init", torch.linspace(0.001, 0.0001, nb_df))
        self.nb_erb = nb_erb
        self.nb_df = nb_df
        self.alpha = alpha

    def erb_feat(self, spec, erb_state):
        erb_state = erb_state + self.erb_init
        power = spec[..., 0] ** 2 + spec[..., 1] ** 2
        band = torch.matmul(power, self.erb_fb)
        db2 = (10.0 * torch.log10(band + 1e-10)).squeeze(1).squeeze(1)
        new_state = db2 * (1 - self.alpha) + erb_state * self.alpha
        feat = ((db2 - new_state) / 40.0).unsqueeze(1).unsqueeze(1)
        return feat, new_state - self.erb_init

    def spec_feat(self, spec, spec_state):
        spec_state = spec_state + self.spec_init
        real = spec[..., : self.nb_df, 0].squeeze(1).squeeze(1)
        imag = spec[..., : self.nb_df, 1].squeeze(1).squeeze(1)
        magnitude = torch.sqrt(real**2 + imag**2)
        new_state = magnitude * (1 - self.alpha) + spec_state * self.alpha
        denom = torch.sqrt(new_state).clamp_min(1e-14)
        feat = torch.stack([real / denom, imag / denom], dim=-1).unsqueeze(1).unsqueeze(1)
        return feat, new_state - self.spec_init


def df_one_step(window, coefs, df_bins):
    real = window[..., :df_bins, 0]
    imag = window[..., :df_bins, 1]
    coef_real = coefs[..., 0]
    coef_imag = coefs[..., 1]
    out_real = torch.sum(real * coef_real - imag * coef_imag, dim=1)
    out_imag = torch.sum(real * coef_imag + imag * coef_real, dim=1)
    return torch.stack([out_real, out_imag], dim=-1)


class StateBank:
    def __init__(self):
        self.specs = []

    def add(self, name, shape):
        self.specs.append((name, tuple(int(size) for size in shape)))

    def total(self):
        count = 0
        for _name, shape in self.specs:
            numel = 1
            for size in shape:
                numel *= size
            count += numel
        return count

    def unpack(self, flat):
        out = {}
        offset = 0
        for name, shape in self.specs:
            numel = 1
            for size in shape:
                numel *= size
            out[name] = flat[offset : offset + numel].reshape(shape)
            offset += numel
        return out

    def pack(self, values):
        return torch.cat([values[name].reshape(-1) for name, _shape in self.specs], dim=0)

    def zeros(self):
        return {name: torch.zeros(shape, dtype=torch.float32) for name, shape in self.specs}


class StreamingHush(nn.Module):
    def __init__(self, df_net, config, norm_alpha: float):
        super().__init__()
        self.model = df_net
        self.sr = config.sr
        self.n_fft = config.fft_size
        self.hop_size = config.hop_size
        self.nb_erb = config.nb_erb
        self.nb_df = config.nb_df
        self.df_order = config.df_order
        self.df_lookahead = config.df_lookahead
        self.conv_lookahead = config.conv_lookahead
        self.freq_bins = config.fft_size // 2 + 1
        self.network_delay = self.conv_lookahead + self.df_lookahead
        self.delay_hops = 1
        self.alpha = round(float(norm_alpha), 3)

        self.stft = StreamingSTFT(self.n_fft, self.hop_size)
        self.feat = FeatureState(df_net.erb_fb.detach().float(), self.nb_erb, self.nb_df, self.alpha)
        self.raw_len = self.df_order + self.network_delay + 4
        self.masked_len = self.df_order + self.network_delay + 4
        self.out_extra = self.df_lookahead + 1
        self._build_causal_layers()
        self._build_state_bank()

    def _build_causal_layers(self):
        enc, dec, dfdec = self.model.enc, self.model.erb_dec, self.model.df_dec
        names = [
            "enc.erb_conv0",
            "enc.erb_conv1",
            "enc.erb_conv2",
            "enc.erb_conv3",
            "enc.df_conv0",
            "enc.df_conv1",
            "erb_dec.conv3p",
            "erb_dec.convt3",
            "erb_dec.conv2p",
            "erb_dec.convt2",
            "erb_dec.conv1p",
            "erb_dec.convt1",
            "erb_dec.conv0p",
            "erb_dec.conv0_out",
            "df_dec.df_convp",
        ]
        frames = 16
        dummy_erb = torch.zeros(1, 1, frames, self.nb_erb)
        dummy_spec = torch.zeros(1, 2, frames, self.nb_df)

        def runner():
            e0, e1, e2, e3, emb, c0, _lsnr = enc(dummy_erb, dummy_spec)
            dec(emb, e3, e2, e1, e0)
            dfdec.df_convp(c0)

        wrapped = wrap_causal_layers(self.model, names, runner)
        self.causal_layers = nn.ModuleDict({key.replace(".", "__"): value for key, value in wrapped.items()})

        def gru_shape(gru_module):
            return (gru_module.gru.num_layers, 1, gru_module.gru.hidden_size)

        self.gru_state_shapes = {
            "enc_emb_gru": gru_shape(enc.emb_gru),
            "erb_dec_emb_gru": gru_shape(dec.emb_gru),
            "df_gru": gru_shape(dfdec.df_gru),
        }

    def _build_state_bank(self):
        bank = StateBank()
        bank.add("raw_buf", (self.raw_len, self.freq_bins, 2))
        bank.add("masked_buf", (self.masked_len, self.freq_bins, 2))
        bank.add("out_buf", (self.out_extra, self.freq_bins, 2))
        bank.add("erb_norm_state", (1, self.nb_erb))
        bank.add("spec_norm_state", (1, self.nb_df))
        for name, layer in self.causal_layers.items():
            if layer.hist > 0:
                bank.add(f"cache__{name}", layer.cache_shape)
        for name, shape in self.gru_state_shapes.items():
            bank.add(f"gru__{name}", shape)
        self.bank = bank
        self.state_size = bank.total()

    def initial_state(self):
        return self.bank.zeros()

    def _run_network(self, feat_erb, feat_spec, state):
        enc, dec, dfdec = self.model.enc, self.model.erb_dec, self.model.df_dec
        cache_updates = {}

        def causal(name):
            key = name.replace(".", "__")
            layer = self.causal_layers[key]
            cache = state.get(f"cache__{key}") if layer.hist > 0 else None
            return layer, cache

        feat_spec = feat_spec.squeeze(1).permute(0, 3, 1, 2)
        layer, cache = causal("enc.erb_conv0")
        e0, cache_updates["enc__erb_conv0"] = layer(feat_erb, cache)
        layer, cache = causal("enc.erb_conv1")
        e1, cache_updates["enc__erb_conv1"] = layer(e0, cache)
        layer, cache = causal("enc.erb_conv2")
        e2, cache_updates["enc__erb_conv2"] = layer(e1, cache)
        layer, cache = causal("enc.erb_conv3")
        e3, cache_updates["enc__erb_conv3"] = layer(e2, cache)
        layer, cache = causal("enc.df_conv0")
        c0, cache_updates["enc__df_conv0"] = layer(feat_spec, cache)
        layer, cache = causal("enc.df_conv1")
        c1, cache_updates["enc__df_conv1"] = layer(c0, cache)

        cemb = c1.permute(0, 2, 3, 1).flatten(2)
        cemb = enc.df_fc_emb(cemb)
        emb = e3.permute(0, 2, 3, 1).flatten(2)
        emb = enc.combine(emb, cemb)

        gru_state = state["gru__enc_emb_gru"]
        emb, gru_state_new = enc.emb_gru(emb, gru_state)
        gru_state2 = state["gru__erb_dec_emb_gru"]
        emb2, gru_state2_new = dec.emb_gru(emb, gru_state2)
        batch, _channels, time, freq = e3.shape
        emb2 = emb2.view(batch, time, freq, -1).permute(0, 3, 1, 2)

        layer, cache = causal("erb_dec.conv3p")
        p3, cache_updates["erb_dec__conv3p"] = layer(e3, cache)
        layer, cache = causal("erb_dec.convt3")
        d3, cache_updates["erb_dec__convt3"] = layer(p3 + emb2, cache)
        layer, cache = causal("erb_dec.conv2p")
        p2, cache_updates["erb_dec__conv2p"] = layer(e2, cache)
        layer, cache = causal("erb_dec.convt2")
        d2, cache_updates["erb_dec__convt2"] = layer(p2 + d3, cache)
        layer, cache = causal("erb_dec.conv1p")
        p1, cache_updates["erb_dec__conv1p"] = layer(e1, cache)
        layer, cache = causal("erb_dec.convt1")
        d1, cache_updates["erb_dec__convt1"] = layer(p1 + d2, cache)
        layer, cache = causal("erb_dec.conv0p")
        p0, cache_updates["erb_dec__conv0p"] = layer(e0, cache)
        layer, cache = causal("erb_dec.conv0_out")
        mask_out, cache_updates["erb_dec__conv0_out"] = layer(p0 + d1, cache)

        gru_state_df = state["gru__df_gru"]
        coef_hidden, gru_state_df_new = dfdec.df_gru(emb, gru_state_df)
        if dfdec.df_skip is not None:
            coef_hidden = coef_hidden + dfdec.df_skip(emb)
        layer, cache = causal("df_dec.df_convp")
        c0p, cache_updates["df_dec__df_convp"] = layer(c0, cache)
        c0p = c0p.permute(0, 2, 3, 1)
        coefs = dfdec.df_out(coef_hidden)
        coefs = coefs.view(coefs.shape[0], coefs.shape[1], self.nb_df, self.df_order * 2)
        coefs = coefs + c0p
        coefs = coefs.unflatten(-1, (-1, 2)).permute(0, 3, 1, 2, 4)
        gru_updates = {
            "enc_emb_gru": gru_state_new,
            "erb_dec_emb_gru": gru_state2_new,
            "df_gru": gru_state_df_new,
        }
        return mask_out, coefs, cache_updates, gru_updates

    def forward(self, speech, analysis_cache, synthesis_cache, state_in):
        state = self.bank.unpack(state_in)
        spec, analysis_cache_out = self.stft.analysis(speech, analysis_cache)
        erb_feat, erb_state_new = self.feat.erb_feat(spec, state["erb_norm_state"].squeeze(0))
        spec_feat, spec_state_new = self.feat.spec_feat(spec, state["spec_norm_state"].squeeze(0))
        state["erb_norm_state"] = erb_state_new.unsqueeze(0)
        state["spec_norm_state"] = spec_state_new.unsqueeze(0)

        mask, df_coefs, cache_updates, gru_updates = self._run_network(erb_feat, spec_feat, state)
        for key, value in cache_updates.items():
            if value is not None and f"cache__{key}" in state:
                state[f"cache__{key}"] = value
        for key, value in gru_updates.items():
            state[f"gru__{key}"] = value

        raw_frame = spec.reshape(1, self.freq_bins, 2)
        raw_buf = torch.cat([state["raw_buf"][1:], raw_frame], dim=0)
        state["raw_buf"] = raw_buf
        raw_current = raw_buf[self.raw_len - 1 - self.conv_lookahead]
        mask_full = torch.matmul(mask.reshape(1, self.nb_erb), self.model.mask.erb_inv_fb)
        masked_frame = raw_current * mask_full.reshape(self.freq_bins, 1)
        masked_buf = torch.cat([state["masked_buf"][1:], masked_frame.unsqueeze(0)], dim=0)
        state["masked_buf"] = masked_buf

        age = self.conv_lookahead - self.df_lookahead
        window = raw_buf[self.raw_len - age - self.df_order : self.raw_len - age].unsqueeze(0)
        coefs = df_coefs.reshape(1, self.df_order, self.nb_df, 2)
        filtered = df_one_step(window, coefs, self.nb_df).squeeze(0)
        out_now = masked_frame.clone()
        out_now[: self.nb_df] = filtered
        out_buf = torch.cat([state["out_buf"][1:], out_now.unsqueeze(0)], dim=0)
        state["out_buf"] = out_buf
        emit = out_buf[self.out_extra - 1 - self.df_lookahead]
        spec_e = emit.reshape(1, 1, 1, self.freq_bins, 2)
        enhanced, synthesis_cache_out = self.stft.synthesis(spec_e, synthesis_cache)
        return enhanced, analysis_cache_out, synthesis_cache_out, self.bank.pack(state)


def load_df_net(checkpoint: Path):
    df_net_se_cls, get_config, get_norm_alpha, enhance = _load_hush_symbols()
    config = get_config()
    wrapper = df_net_se_cls(config).eval()
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    wrapper.model.load_state_dict(state, strict=True)
    wrapper.model.eval()
    alpha = get_norm_alpha(config.sr, config.hop_size, config.norm_tau)
    return wrapper, config, alpha, enhance


def load_test_waveform(sample_rate: int, hop_size: int) -> torch.Tensor:
    wav_path = REPO_ROOT / "public" / "wavs" / "zh.wav"
    num_frames = 200
    target = num_frames * hop_size
    if wav_path.exists():
        _, data = avioflow.load(str(wav_path), output_sample_rate=sample_rate, output_num_channels=1)
        wav = torch.from_numpy(data)
        if wav.shape[1] == 0:
            raise ValueError(f"Empty audio: {wav_path}")
        if wav.shape[1] < target:
            wav = wav.repeat(1, target // wav.shape[1] + 1)
        return wav[:, :target].to(torch.float32).contiguous()
    torch.manual_seed(20260815)
    return (torch.rand(1, target) * 2.0 - 1.0).to(torch.float32)


def run_streaming(stream: StreamingHush, waveform: torch.Tensor) -> torch.Tensor:
    hop = stream.hop_size
    analysis_cache = torch.zeros(1, hop)
    synthesis_cache = torch.zeros(1, hop)
    state = stream.bank.pack(stream.initial_state())
    outputs = []
    with torch.no_grad():
        for offset in range(0, waveform.shape[1] - hop + 1, hop):
            speech = waveform[:, offset : offset + hop]
            enhanced, analysis_cache, synthesis_cache, state = stream(
                speech, analysis_cache, synthesis_cache, state
            )
            outputs.append(enhanced)
    return torch.cat(outputs, dim=1)


def measure_delay(stream: StreamingHush, waveform: torch.Tensor, enhance, reference_model) -> float:
    with torch.no_grad():
        reference = enhance(reference_model, waveform, pad_delay=True)
    streaming = run_streaming(stream, waveform)
    hop = stream.hop_size
    best_shift = 0
    best_diff = float("inf")
    limit = min(4, streaming.shape[1] // hop)
    for shift in range(limit + 1):
        offset = shift * hop
        count = min(reference.shape[1], streaming.shape[1] - offset)
        if count <= hop:
            continue
        # Ignore the flush tail, where libdf zero-padding and our hop grid differ.
        count = max(hop, count - stream.n_fft)
        diff = float(torch.max(torch.abs(reference[:, :count] - streaming[:, offset : offset + count])))
        print(f"libdf shift {shift} hop max abs diff: {diff:.8g}")
        if diff < best_diff:
            best_diff = diff
            best_shift = shift
    stream.delay_hops = best_shift
    return best_diff


def verify_onnx(stream: StreamingHush, output_path: str, waveform: torch.Tensor):
    hop = stream.hop_size
    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    torch_state = [
        torch.zeros(1, hop),
        torch.zeros(1, hop),
        stream.bank.pack(stream.initial_state()),
    ]
    ort_state = [value.numpy().copy() for value in torch_state]
    wav_diff = 0.0
    state_diff = 0.0
    with torch.no_grad():
        for offset in range(0, waveform.shape[1] - hop + 1, hop):
            speech = waveform[:, offset : offset + hop]
            torch_out = stream(speech, *torch_state)
            feeds = {
                "speech": speech.numpy(),
                "analysis_cache": ort_state[0],
                "synthesis_cache": ort_state[1],
                "state_in": ort_state[2],
            }
            ort_out = session.run(None, feeds)
            wav_diff = max(wav_diff, float(np.max(np.abs(torch_out[0].numpy() - ort_out[0]))))
            for torch_value, ort_value in zip(torch_out[1:], ort_out[1:]):
                state_diff = max(state_diff, float(np.max(np.abs(torch_value.numpy() - ort_value))))
            torch_state = [value.detach().clone() for value in torch_out[1:]]
            ort_state = [value.copy() for value in ort_out[1:]]
    return wav_diff, state_diff


def export_onnx(stream: StreamingHush, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hop = stream.hop_size
    dummy = (
        torch.zeros(1, hop),
        torch.zeros(1, hop),
        torch.zeros(1, hop),
        stream.bank.pack(stream.initial_state()),
    )
    torch.onnx.export(
        stream,
        dummy,
        str(output_path),
        input_names=["speech", "analysis_cache", "synthesis_cache", "state_in"],
        output_names=["enhanced", "analysis_cache_out", "synthesis_cache_out", "state_out"],
        dynamic_axes={
            "speech": {0: "batch"},
            "analysis_cache": {0: "batch"},
            "synthesis_cache": {0: "batch"},
            "enhanced": {0: "batch"},
            "analysis_cache_out": {0: "batch"},
            "synthesis_cache_out": {0: "batch"},
        },
        opset_version=OPSET_VERSION,
        dynamo=False,
    )


def add_metadata(output_path: Path, stream: StreamingHush):
    model = onnx.load(str(output_path))
    metadata = {
        "model_type": MODEL_TYPE,
        "sample_rate": str(stream.sr),
        "frame_length": str(stream.n_fft),
        "frame_shift": str(stream.hop_size),
        "state_size": str(stream.state_size),
        "streaming": "1",
        "delay_hops": str(stream.delay_hops),
    }
    del model.metadata_props[:]
    for key, value in metadata.items():
        item = model.metadata_props.add()
        item.key = key
        item.value = value
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))


def quantize_onnx_model(input_path: Path, output_path: Path):
    model = onnx.load(str(input_path))
    nodes_to_exclude = []
    preprocess_keywords = (
        "stft_analysis_real",
        "stft_analysis_imag",
        "stft_synthesis_real",
        "stft_synthesis_imag",
        "stft_window",
        "stft_wnorm",
        "stft_inv_wnorm",
        "erb_fb",
        "erb_inv_fb",
        "erb_init",
        "spec_init",
    )
    preprocess_inits = [
        init.name
        for init in model.graph.initializer
        if any(keyword in init.name.lower() for keyword in preprocess_keywords)
    ]
    for node in model.graph.node:
        node_name = node.name.lower()
        if any(name in preprocess_inits for name in node.input):
            nodes_to_exclude.append(node.name)
            continue
        if any(keyword in node_name for keyword in preprocess_keywords):
            nodes_to_exclude.append(node.name)
            continue
        if node.op_type == "Conv":
            group = next((attr.i for attr in node.attribute if attr.name == "group"), 1)
            if group != 1:
                nodes_to_exclude.append(node.name)
    nodes_to_exclude = sorted(set(nodes_to_exclude))
    print(f"Excluding {len(nodes_to_exclude)} nodes from int8 quantization")
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QUInt8,
        nodes_to_exclude=nodes_to_exclude,
        per_channel=False,
        reduce_range=False,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Export streaming hush_dfnet_16k ONNX")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--onnx-path", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--verify", type=int, default=1)
    parser.add_argument("--quantize", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    wrapper, config, alpha, enhance = load_df_net(args.checkpoint)
    stream = StreamingHush(wrapper.model, config, alpha).eval()
    print(
        f"state_size={stream.state_size} hop={stream.hop_size} "
        f"alpha={stream.alpha} network_delay={stream.network_delay}"
    )
    waveform = load_test_waveform(stream.sr, stream.hop_size)
    libdf_diff = float("nan")
    if args.verify:
        libdf_diff = measure_delay(stream, waveform, enhance, wrapper)
        print(f"selected delay_hops={stream.delay_hops} libdf max abs diff={libdf_diff:.8g}")
        if not (libdf_diff < 1e-3):
            raise SystemExit(f"libdf alignment diff {libdf_diff:.8g} exceeds 1e-3")

    output_path = args.onnx_path if args.onnx_path.is_absolute() else REPO_ROOT / args.onnx_path
    export_onnx(stream, output_path)
    add_metadata(output_path, stream)
    size = output_path.stat().st_size
    print(f"Exported: {output_path} ({size / 1024 / 1024:.2f} MB)")

    onnx_wav_diff = onnx_state_diff = float("nan")
    if args.verify:
        onnx_wav_diff, onnx_state_diff = verify_onnx(stream, str(output_path), waveform)
        print(
            f"streaming-PyTorch vs ONNX max abs diff: "
            f"waveform={onnx_wav_diff:.8g}, state={onnx_state_diff:.8g}"
        )
        if not (onnx_wav_diff < 1e-4 and onnx_state_diff < 1e-4):
            raise SystemExit("ONNX parity exceeds 1e-4")

    if args.quantize:
        quantized_path = output_path.with_name(output_path.stem + ".int8.onnx")
        quantize_onnx_model(output_path, quantized_path)
        add_metadata(quantized_path, stream)
        qsize = quantized_path.stat().st_size
        print(f"Quantized: {quantized_path} ({qsize / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
