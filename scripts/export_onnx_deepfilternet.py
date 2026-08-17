#!/usr/bin/env python3
"""Export streaming (frame-by-frame, causal) ONNX graphs for DeepFilterNet v1/v2/v3.

Structural template: scripts/export_onnx_dpdfnet.py (same waveform-domain
streaming I/O contract: speech / analysis_cache / synthesis_cache / state).
"""

import argparse
import math
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic

OPSET_VERSION = 18
REPO_ROOT = Path(__file__).resolve().parents[1]


def _patch_torchaudio():
    import types

    import torchaudio

    if not hasattr(torchaudio, "backend") or not hasattr(torchaudio.backend, "common"):
        backend_mod = types.ModuleType("torchaudio.backend")
        common_mod = types.ModuleType("torchaudio.backend.common")
        common_mod.AudioMetaData = getattr(torchaudio, "AudioMetaData", object)
        sys.modules["torchaudio.backend"] = backend_mod
        sys.modules["torchaudio.backend.common"] = common_mod
        torchaudio.backend = backend_mod


_patch_torchaudio()

from df.config import config as df_config  # noqa: E402
from df.model import ModelParams  # noqa: E402
from libdf import DF  # noqa: E402

VARIANTS = {
    1: {"zip": "DeepFilterNet.zip", "model_type": "deepfilternet_denoise", "suffix": ""},
    2: {"zip": "DeepFilterNet2.zip", "model_type": "deepfilternet2_denoise", "suffix": "2"},
    3: {"zip": "DeepFilterNet3.zip", "model_type": "deepfilternet3_denoise", "suffix": "3"},
}


# --------------------------------------------------------------------------
# STFT (ported from scripts/export_onnx_dpdfnet.py; DeepFilterNet spec is
# [B,1,1,F,2] instead of DPDFNet's [B,1,F,2]).
# --------------------------------------------------------------------------
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
        spec = spec.unsqueeze(1).unsqueeze(1)  # [B,1,1,F,2]
        analysis_cache_out = speech
        return spec, analysis_cache_out

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


# --------------------------------------------------------------------------
# Causal conv wrapper.
#
# Every Conv2dNormAct/ConvTranspose2dNormAct/convkxf Sequential that has a
# non-trivial time kernel prepends an nn.ConstantPad2d((fl,fr,tb,ta)) before
# the actual Conv2d/ConvTranspose2d (which itself uses zero time-padding).
# For a feed-forward layer, treating the whole tb+ta as pure *history* (a
# ring buffer of that many past real frames) instead of tb real-history +
# ta zero-padded lookahead is mathematically identical -- it only shifts the
# logical timestamp of the result forward by ta frames. We correct for this
# globally (see StreamingDeepFilterNet).
# --------------------------------------------------------------------------
class CausalConv(nn.Module):
    def __init__(self, seq: nn.Sequential, cache_shape):
        super().__init__()
        children = list(seq.children())
        if len(children) > 0 and isinstance(children[0], nn.ConstantPad2d):
            pad = children[0].padding  # (freq_l, freq_r, time_before, time_after)
            self.hist = int(pad[2] + pad[3])
            self.rest = nn.Sequential(*children[1:])
        else:
            self.hist = 0
            self.rest = nn.Sequential(*children)
        self.cache_shape = tuple(cache_shape)  # (1, C, hist, F)

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
    for h in hooks:
        h.remove()
    return shapes


def wrap_causal_layers(root, names, runner):
    shapes = capture_input_shapes(root, names, runner)
    named = dict(root.named_modules())
    wrapped = {}
    for name in names:
        seq = named[name]
        b, c, t, f = shapes[name]
        children = list(seq.children())
        hist = 0
        if len(children) > 0 and isinstance(children[0], nn.ConstantPad2d):
            pad = children[0].padding
            hist = int(pad[2] + pad[3])
        wrapped[name] = CausalConv(seq, (1, c, hist, f))
    return wrapped


# --------------------------------------------------------------------------
# ERB + unit-norm feature extraction (mirrors libDF Rust erb()/erb_norm()/
# unit_norm()).
# --------------------------------------------------------------------------
class FeatureState(nn.Module):
    def __init__(self, erb_fb_mat: torch.Tensor, nb_erb: int, nb_df: int, alpha: float):
        super().__init__()
        self.register_buffer("erb_fb", erb_fb_mat)  # [F, nb_erb]
        self.nb_erb = nb_erb
        self.nb_df = nb_df
        self.alpha = alpha

    def erb_feat(self, spec, erb_state):
        # spec: [B,1,1,F,2]; erb_state: [B, nb_erb]
        power = spec[..., 0] ** 2 + spec[..., 1] ** 2  # [B,1,1,F]
        band = torch.matmul(power, self.erb_fb)  # [B,1,1,nb_erb]
        db2 = (10.0 * torch.log10(band + 1e-10)).squeeze(1).squeeze(1)  # [B,nb_erb]
        new_state = db2 * (1 - self.alpha) + erb_state * self.alpha
        feat = ((db2 - new_state) / 40.0).unsqueeze(1).unsqueeze(1)  # [B,1,1,nb_erb]
        return feat, new_state

    def spec_feat(self, spec, spec_state):
        # spec: [B,1,1,F,2]; spec_state: [B, nb_df]
        re = spec[..., : self.nb_df, 0].squeeze(1).squeeze(1)
        im = spec[..., : self.nb_df, 1].squeeze(1).squeeze(1)
        mag = torch.sqrt(re**2 + im**2)
        new_state = mag * (1 - self.alpha) + spec_state * self.alpha
        denom = torch.sqrt(new_state).clamp_min(1e-14)
        re2 = re / denom
        im2 = im / denom
        feat = torch.stack([re2, im2], dim=-1).unsqueeze(1).unsqueeze(1)  # [B,1,1,nb_df,2]
        return feat, new_state


def df_one_step(window, coefs, df_bins):
    """window: [B,O,F,2] (oldest..newest); coefs: [B,O,df_bins,2] -> [B,df_bins,2]."""
    sre = window[..., :df_bins, 0]
    sim = window[..., :df_bins, 1]
    cre = coefs[..., 0]
    cim = coefs[..., 1]
    outr = torch.sum(sre * cre - sim * cim, dim=1)
    outi = torch.sum(sre * cim + sim * cre, dim=1)
    return torch.stack([outr, outi], dim=-1)


class StateBank:
    def __init__(self):
        self.specs = []

    def add(self, name, shape):
        self.specs.append((name, tuple(int(s) for s in shape)))

    def total(self):
        n = 0
        for _, shape in self.specs:
            m = 1
            for s in shape:
                m *= s
            n += m
        return n

    def unpack(self, flat):
        out = {}
        off = 0
        for name, shape in self.specs:
            numel = 1
            for s in shape:
                numel *= s
            out[name] = flat[off : off + numel].reshape(shape)
            off += numel
        return out

    def pack(self, d):
        return torch.cat([d[name].reshape(-1) for name, _ in self.specs], dim=0)

    def zeros(self):
        return {name: torch.zeros(shape, dtype=torch.float32) for name, shape in self.specs}


# --------------------------------------------------------------------------
# Shared streaming pipeline
# --------------------------------------------------------------------------
class StreamingDeepFilterNet(nn.Module):
    def __init__(self, model, variant: int):
        super().__init__()
        self.model = model
        self.variant = variant
        p = ModelParams()
        self.sr = p.sr
        self.n_fft = p.fft_size
        self.hop_size = p.hop_size
        self.nb_erb = p.nb_erb
        self.nb_df = p.nb_df
        self.df_order = p.df_order
        self.df_lookahead = p.df_lookahead
        self.conv_lookahead = p.conv_lookahead
        self.freq_bins = p.fft_size // 2 + 1
        self.total_delay = self.conv_lookahead + self.df_lookahead

        alpha = math.exp(-p.hop_size / (p.sr * p.norm_tau))
        alpha = round(alpha, 3)
        self.alpha = alpha

        from df.modules import erb_fb

        df_state = DF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size, nb_bands=p.nb_erb,
                      min_nb_erb_freqs=p.min_nb_freqs)
        widths = df_state.erb_widths()
        erb_fb_mat = erb_fb(widths, p.sr, inverse=False)
        self.stft = StreamingSTFT(p.fft_size, p.hop_size)
        self.feat = FeatureState(erb_fb_mat, p.nb_erb, p.nb_df, alpha)

        self.raw_len = self.df_order + self.total_delay + 4
        self.masked_len = self.df_order + self.total_delay + 4
        self.out_extra = self.df_lookahead + 1

        self.uses_masked_spec_for_df = variant in (1, 2)

        self._build_causal_layers()
        self._build_state_bank()

    def _build_causal_layers(self):
        raise NotImplementedError

    def _run_network(self, feat_erb, feat_spec, st):
        raise NotImplementedError

    def _build_state_bank(self):
        bank = StateBank()
        bank.add("raw_buf", (self.raw_len, self.freq_bins, 2))
        bank.add("masked_buf", (self.masked_len, self.freq_bins, 2))
        bank.add("out_buf", (self.out_extra, self.freq_bins, 2))
        bank.add("erb_norm_state", (1, self.nb_erb))
        bank.add("spec_norm_state", (1, self.nb_df))
        for name, layer in self.causal_layers.items():
            if layer.hist > 0:
                # Zero-history layers have a degenerate (size-0) cache that
                # must not be part of the flat state vector: PyTorch's ONNX
                # shape-inference for Reshape divides by the product of
                # known dims when resolving a "-1" dim, and a literal 0 in
                # that product triggers a native SIGFPE (division by zero)
                # during export.
                bank.add(f"cache__{name}", layer.cache_shape)
        for name, shape in self.gru_state_shapes.items():
            bank.add(f"gru__{name}", shape)
        self.bank = bank
        self.state_size = bank.total()

    def initial_state(self):
        st = self.bank.zeros()
        # libDF's erb_norm/unit_norm running-mean states are not zero-
        # initialized; they start from a linspace ramp (see
        # libDF/src/lib.rs MEAN_NORM_INIT / UNIT_NORM_INIT).
        erb_init = torch.linspace(-60.0, -90.0, self.nb_erb).unsqueeze(0)
        spec_init = torch.linspace(0.001, 0.0001, self.nb_df).unsqueeze(0)
        st["erb_norm_state"] = erb_init
        st["spec_norm_state"] = spec_init
        return st

    def forward(self, speech, analysis_cache, synthesis_cache, state_in):
        st = self.bank.unpack(state_in)

        spec, analysis_cache_out = self.stft.analysis(speech, analysis_cache)

        erb_feat, erb_state_new = self.feat.erb_feat(spec, st["erb_norm_state"].squeeze(0))
        spec_feat, spec_state_new = self.feat.spec_feat(spec, st["spec_norm_state"].squeeze(0))
        st["erb_norm_state"] = erb_state_new.unsqueeze(0)
        st["spec_norm_state"] = spec_state_new.unsqueeze(0)

        m, df_coefs, df_alpha, cache_updates, gru_updates = self._run_network(
            erb_feat, spec_feat, st
        )
        for k, v in cache_updates.items():
            if v is not None and f"cache__{k}" in st:
                st[f"cache__{k}"] = v
        for k, v in gru_updates.items():
            st[f"gru__{k}"] = v

        raw_frame = spec.reshape(1, self.freq_bins, 2)
        raw_buf = torch.cat([st["raw_buf"][1:], raw_frame], dim=0)
        st["raw_buf"] = raw_buf

        raw_q = raw_buf[self.raw_len - 1 - self.conv_lookahead]
        m_flat = m.reshape(1, self.nb_erb)
        erb_inv_fb = self._erb_inv_fb()
        mask_full = torch.matmul(m_flat, erb_inv_fb)  # [1,F]
        masked_frame = raw_q * mask_full.reshape(self.freq_bins, 1)

        masked_buf = torch.cat([st["masked_buf"][1:], masked_frame.unsqueeze(0)], dim=0)
        st["masked_buf"] = masked_buf

        if self.uses_masked_spec_for_df:
            window = masked_buf[-self.df_order :].unsqueeze(0)
            center = masked_buf[-1 - self.df_lookahead]
        else:
            age0 = self.conv_lookahead - self.df_lookahead
            window = raw_buf[self.raw_len - age0 - self.df_order : self.raw_len - age0].unsqueeze(
                0
            )
            center = masked_frame

        coefs = df_coefs.reshape(1, self.df_order, self.nb_df, 2)
        filtered = df_one_step(window, coefs, self.nb_df).squeeze(0)

        out_now = center.clone()
        if df_alpha is not None:
            a = df_alpha.reshape(1, 1)
            out_now[: self.nb_df] = filtered * a + center[: self.nb_df] * (1 - a)
        else:
            out_now[: self.nb_df] = filtered

        out_buf = torch.cat([st["out_buf"][1:], out_now.unsqueeze(0)], dim=0)
        st["out_buf"] = out_buf

        extra_delay = 0 if self.uses_masked_spec_for_df else self.df_lookahead
        emit = out_buf[self.out_extra - 1 - extra_delay]
        spec_e = emit.reshape(1, 1, 1, self.freq_bins, 2)

        enhanced, synthesis_cache_out = self.stft.synthesis(spec_e, synthesis_cache)

        state_out = self.bank.pack(st)
        return enhanced, analysis_cache_out, synthesis_cache_out, state_out

    def _erb_inv_fb(self):
        return self.model.mask.erb_inv_fb


# --------------------------------------------------------------------------
# Variant 1: DeepFilterNet (df/deepfilternet.py)
# --------------------------------------------------------------------------
class StreamingV1(StreamingDeepFilterNet):
    def _build_causal_layers(self):
        m = self.model
        enc, dec, dfdec = m.enc, m.erb_dec, m.df_dec
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
        T = 16
        dummy_erb = torch.zeros(1, 1, T, self.nb_erb)
        dummy_spec = torch.zeros(1, 2, T, self.nb_df)

        def runner():
            e0, e1, e2, e3, emb, c0, lsnr = enc(dummy_erb, dummy_spec)
            dec(emb, e3, e2, e1, e0)
            dfdec.df_convp(c0)

        wrapped = wrap_causal_layers(m, names, runner)
        self.causal_layers = nn.ModuleDict({k.replace(".", "__"): v for k, v in wrapped.items()})
        self.gru_state_shapes = {
            "enc_emb_gru": tuple(enc.emb_gru.get_h0(1).shape),
            "df_gru": tuple(dfdec.df_gru.get_h0(1).shape),
        }

    def _run_network(self, feat_erb, feat_spec, st):
        m, enc, dec, dfdec = self.model, self.model.enc, self.model.erb_dec, self.model.df_dec
        cache_updates = {}

        def cl(name):
            key = name.replace(".", "__")
            layer = self.causal_layers[key]
            cache = st.get(f"cache__{key}") if layer.hist > 0 else None
            return layer, cache

        feat_spec_c = feat_spec.transpose(1, 4).squeeze(4)  # [B,2,1,nb_df]

        layer, cache = cl("enc.erb_conv0")
        e0, cache_updates["enc__erb_conv0"] = layer(feat_erb, cache)
        layer, cache = cl("enc.erb_conv1")
        e1, cache_updates["enc__erb_conv1"] = layer(e0, cache)
        layer, cache = cl("enc.erb_conv2")
        e2, cache_updates["enc__erb_conv2"] = layer(e1, cache)
        layer, cache = cl("enc.erb_conv3")
        e3, cache_updates["enc__erb_conv3"] = layer(e2, cache)
        layer, cache = cl("enc.df_conv0")
        c0, cache_updates["enc__df_conv0"] = layer(feat_spec_c, cache)
        layer, cache = cl("enc.df_conv1")
        c1, cache_updates["enc__df_conv1"] = layer(c0, cache)

        b, _, t, _ = e0.shape
        cemb = c1.permute(2, 0, 1, 3).reshape(t, b, -1)
        cemb = enc.df_fc_emb(cemb)
        emb = e3.permute(2, 0, 1, 3).reshape(t, b, -1)
        emb = emb + cemb

        gru_state = st["gru__enc_emb_gru"]
        emb, gru_state_new = enc.emb_gru(emb, gru_state)
        emb_bt = emb.transpose(0, 1)

        b2, t2, f8 = e3.shape[0], e3.shape[2], e3.shape[3]
        emb2 = dec.fc_emb(emb_bt)
        emb2 = emb2.view(b2, t2, -1, f8).transpose(1, 2)

        layer, cache = cl("erb_dec.conv3p")
        p3, cache_updates["erb_dec__conv3p"] = layer(e3, cache)
        s3 = p3 + emb2
        layer, cache = cl("erb_dec.convt3")
        d3, cache_updates["erb_dec__convt3"] = layer(s3, cache)

        layer, cache = cl("erb_dec.conv2p")
        p2, cache_updates["erb_dec__conv2p"] = layer(e2, cache)
        s2 = p2 + d3
        layer, cache = cl("erb_dec.convt2")
        d2, cache_updates["erb_dec__convt2"] = layer(s2, cache)

        layer, cache = cl("erb_dec.conv1p")
        p1, cache_updates["erb_dec__conv1p"] = layer(e1, cache)
        s1 = p1 + d2
        layer, cache = cl("erb_dec.convt1")
        d1, cache_updates["erb_dec__convt1"] = layer(s1, cache)

        layer, cache = cl("erb_dec.conv0p")
        p0, cache_updates["erb_dec__conv0p"] = layer(e0, cache)
        s0 = p0 + d1
        layer, cache = cl("erb_dec.conv0_out")
        mask_out, cache_updates["erb_dec__conv0_out"] = layer(s0, cache)

        gru_state_df = st["gru__df_gru"]
        c_gru, gru_state_df_new = dfdec.df_gru(emb, gru_state_df)
        layer, cache = cl("df_dec.df_convp")
        c0p, cache_updates["df_dec__df_convp"] = layer(c0, cache)
        c0p = c0p.transpose(1, 2)  # [B,T,O*2,F]
        c_gru_bt = c_gru.transpose(0, 1)  # [B,T,H]
        alpha = dfdec.df_fc_a(c_gru_bt)
        c_out = dfdec.df_fc_out(c_gru_bt)
        c_out = c_out.view(b2, t2, self.df_order * 2, self.nb_df)
        c_out = c_out.add(c0p).view(b2, t2, self.df_order, 2, self.nb_df).transpose(3, 4)

        gru_updates = {"enc_emb_gru": gru_state_new, "df_gru": gru_state_df_new}

        # mask_out: [B,1,T,nb_erb]; alpha: [B,T,1]
        return mask_out, c_out, alpha, cache_updates, gru_updates


# --------------------------------------------------------------------------
# Variant 2 / 3: DeepFilterNet2 / DeepFilterNet3
# --------------------------------------------------------------------------
class StreamingV23(StreamingDeepFilterNet):
    def _build_causal_layers(self):
        m = self.model
        enc, dec, dfdec = m.enc, m.erb_dec, m.df_dec
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
        T = 16
        dummy_erb = torch.zeros(1, 1, T, self.nb_erb)
        dummy_spec = torch.zeros(1, 2, T, self.nb_df)

        def runner():
            e0, e1, e2, e3, emb, c0, lsnr = enc(dummy_erb, dummy_spec)
            dec(emb, e3, e2, e1, e0)
            dfdec.df_convp(c0)

        wrapped = wrap_causal_layers(m, names, runner)
        self.causal_layers = nn.ModuleDict({k.replace(".", "__"): v for k, v in wrapped.items()})

        def gru_h0(gru_module):
            return torch.zeros(gru_module.gru.num_layers, 1, gru_module.gru.hidden_size).shape

        self.gru_state_shapes = {
            "enc_emb_gru": tuple(gru_h0(enc.emb_gru)),
            "erb_dec_emb_gru": tuple(gru_h0(dec.emb_gru)),
            "df_gru": tuple(gru_h0(dfdec.df_gru)),
        }

    def _run_network(self, feat_erb, feat_spec, st):
        m, enc, dec, dfdec = self.model, self.model.enc, self.model.erb_dec, self.model.df_dec
        cache_updates = {}

        def cl(name):
            key = name.replace(".", "__")
            layer = self.causal_layers[key]
            cache = st.get(f"cache__{key}") if layer.hist > 0 else None
            return layer, cache

        feat_spec_c = feat_spec.squeeze(1).permute(0, 3, 1, 2)  # [B,2,T,F]

        layer, cache = cl("enc.erb_conv0")
        e0, cache_updates["enc__erb_conv0"] = layer(feat_erb, cache)
        layer, cache = cl("enc.erb_conv1")
        e1, cache_updates["enc__erb_conv1"] = layer(e0, cache)
        layer, cache = cl("enc.erb_conv2")
        e2, cache_updates["enc__erb_conv2"] = layer(e1, cache)
        layer, cache = cl("enc.erb_conv3")
        e3, cache_updates["enc__erb_conv3"] = layer(e2, cache)
        layer, cache = cl("enc.df_conv0")
        c0, cache_updates["enc__df_conv0"] = layer(feat_spec_c, cache)
        layer, cache = cl("enc.df_conv1")
        c1, cache_updates["enc__df_conv1"] = layer(c0, cache)

        cemb = c1.permute(0, 2, 3, 1).flatten(2)
        cemb = enc.df_fc_emb(cemb)
        emb = e3.permute(0, 2, 3, 1).flatten(2)
        emb = enc.combine(emb, cemb)

        gru_state = st["gru__enc_emb_gru"]
        emb, gru_state_new = enc.emb_gru(emb, gru_state)

        gru_state2 = st["gru__erb_dec_emb_gru"]
        emb2, gru_state2_new = dec.emb_gru(emb, gru_state2)
        emb2 = dec.fc_emb(emb2)
        b2, t2, f8 = e3.shape[0], e3.shape[2], e3.shape[3]
        emb2 = emb2.view(b2, t2, f8, -1).permute(0, 3, 1, 2)

        layer, cache = cl("erb_dec.conv3p")
        p3, cache_updates["erb_dec__conv3p"] = layer(e3, cache)
        s3 = p3 + emb2
        layer, cache = cl("erb_dec.convt3")
        d3, cache_updates["erb_dec__convt3"] = layer(s3, cache)

        layer, cache = cl("erb_dec.conv2p")
        p2, cache_updates["erb_dec__conv2p"] = layer(e2, cache)
        s2 = p2 + d3
        layer, cache = cl("erb_dec.convt2")
        d2, cache_updates["erb_dec__convt2"] = layer(s2, cache)

        layer, cache = cl("erb_dec.conv1p")
        p1, cache_updates["erb_dec__conv1p"] = layer(e1, cache)
        s1 = p1 + d2
        layer, cache = cl("erb_dec.convt1")
        d1, cache_updates["erb_dec__convt1"] = layer(s1, cache)

        layer, cache = cl("erb_dec.conv0p")
        p0, cache_updates["erb_dec__conv0p"] = layer(e0, cache)
        s0 = p0 + d1
        layer, cache = cl("erb_dec.conv0_out")
        mask_out, cache_updates["erb_dec__conv0_out"] = layer(s0, cache)

        gru_state_df = st["gru__df_gru"]
        c_gru, gru_state_df_new = dfdec.df_gru(emb, gru_state_df)
        if dfdec.df_skip is not None:
            c_gru = c_gru + dfdec.df_skip(emb)

        layer, cache = cl("df_dec.df_convp")
        c0p, cache_updates["df_dec__df_convp"] = layer(c0, cache)
        c0p = c0p.permute(0, 2, 3, 1)  # [B,T,F,O*2]

        c_out = dfdec.df_out(c_gru)  # [B,T,F*O*2]
        c_out = c_out.view(c_out.shape[0], c_out.shape[1], self.nb_df, self.df_order * 2)
        c_out = c_out + c0p  # [B,T,F,O*2]
        # DfOutputReshapeMF: [B,T,F,O*2] -> [B,O,T,F,2]
        c_out = c_out.unflatten(-1, (-1, 2)).permute(0, 3, 1, 2, 4)

        gru_updates = {
            "enc_emb_gru": gru_state_new,
            "erb_dec_emb_gru": gru_state2_new,
            "df_gru": gru_state_df_new,
        }
        return mask_out, c_out, None, cache_updates, gru_updates


class StreamingV3(StreamingV23):
    def _run_network(self, feat_erb, feat_spec, st):
        m, enc, dec, dfdec = self.model, self.model.enc, self.model.erb_dec, self.model.df_dec
        cache_updates = {}

        def cl(name):
            key = name.replace(".", "__")
            layer = self.causal_layers[key]
            cache = st.get(f"cache__{key}") if layer.hist > 0 else None
            return layer, cache

        feat_spec_c = feat_spec.squeeze(1).permute(0, 3, 1, 2)

        layer, cache = cl("enc.erb_conv0")
        e0, cache_updates["enc__erb_conv0"] = layer(feat_erb, cache)
        layer, cache = cl("enc.erb_conv1")
        e1, cache_updates["enc__erb_conv1"] = layer(e0, cache)
        layer, cache = cl("enc.erb_conv2")
        e2, cache_updates["enc__erb_conv2"] = layer(e1, cache)
        layer, cache = cl("enc.erb_conv3")
        e3, cache_updates["enc__erb_conv3"] = layer(e2, cache)
        layer, cache = cl("enc.df_conv0")
        c0, cache_updates["enc__df_conv0"] = layer(feat_spec_c, cache)
        layer, cache = cl("enc.df_conv1")
        c1, cache_updates["enc__df_conv1"] = layer(c0, cache)

        cemb = c1.permute(0, 2, 3, 1).flatten(2)
        cemb = enc.df_fc_emb(cemb)
        emb = e3.permute(0, 2, 3, 1).flatten(2)
        emb = enc.combine(emb, cemb)

        gru_state = st["gru__enc_emb_gru"]
        emb, gru_state_new = enc.emb_gru(emb, gru_state)

        gru_state2 = st["gru__erb_dec_emb_gru"]
        emb2, gru_state2_new = dec.emb_gru(emb, gru_state2)
        b2, t2, f8 = e3.shape[0], e3.shape[2], e3.shape[3]
        emb2 = emb2.view(b2, t2, f8, -1).permute(0, 3, 1, 2)

        layer, cache = cl("erb_dec.conv3p")
        p3, cache_updates["erb_dec__conv3p"] = layer(e3, cache)
        s3 = p3 + emb2
        layer, cache = cl("erb_dec.convt3")
        d3, cache_updates["erb_dec__convt3"] = layer(s3, cache)

        layer, cache = cl("erb_dec.conv2p")
        p2, cache_updates["erb_dec__conv2p"] = layer(e2, cache)
        s2 = p2 + d3
        layer, cache = cl("erb_dec.convt2")
        d2, cache_updates["erb_dec__convt2"] = layer(s2, cache)

        layer, cache = cl("erb_dec.conv1p")
        p1, cache_updates["erb_dec__conv1p"] = layer(e1, cache)
        s1 = p1 + d2
        layer, cache = cl("erb_dec.convt1")
        d1, cache_updates["erb_dec__convt1"] = layer(s1, cache)

        layer, cache = cl("erb_dec.conv0p")
        p0, cache_updates["erb_dec__conv0p"] = layer(e0, cache)
        s0 = p0 + d1
        layer, cache = cl("erb_dec.conv0_out")
        mask_out, cache_updates["erb_dec__conv0_out"] = layer(s0, cache)

        gru_state_df = st["gru__df_gru"]
        c_gru, gru_state_df_new = dfdec.df_gru(emb, gru_state_df)
        if dfdec.df_skip is not None:
            c_gru = c_gru + dfdec.df_skip(emb)

        layer, cache = cl("df_dec.df_convp")
        c0p, cache_updates["df_dec__df_convp"] = layer(c0, cache)
        c0p = c0p.permute(0, 2, 3, 1)

        c_out = dfdec.df_out(c_gru)
        c_out = c_out.view(c_out.shape[0], c_out.shape[1], self.nb_df, self.df_order * 2)
        c_out = c_out + c0p
        c_out = c_out.unflatten(-1, (-1, 2)).permute(0, 3, 1, 2, 4)

        gru_updates = {
            "enc_emb_gru": gru_state_new,
            "erb_dec_emb_gru": gru_state2_new,
            "df_gru": gru_state_df_new,
        }
        return mask_out, c_out, None, cache_updates, gru_updates


VARIANT_CLASSES = {1: StreamingV1, 2: StreamingV23, 3: StreamingV3}


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def unzip_model(zip_path: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    # find the dir with config.ini
    for p in dest_dir.rglob("config.ini"):
        return p.parent
    raise FileNotFoundError(f"No config.ini found after extracting {zip_path}")


def load_variant_model(variant: int, model_dir: str = None):
    from df.checkpoint import load_model as df_ckpt_load_model
    from df.model import init_model

    info = VARIANTS[variant]
    if model_dir is None:
        zip_path = REPO_ROOT / "debug" / "DeepFilterNet" / "models" / info["zip"]
        cache_dir = Path(tempfile.gettempdir()) / "dfn_export" / f"DeepFilterNet{info['suffix']}"
        if not (cache_dir.exists() and any(cache_dir.rglob("config.ini"))):
            model_dir = unzip_model(zip_path, cache_dir)
        else:
            model_dir = next(cache_dir.rglob("config.ini")).parent
    model_dir = Path(model_dir)

    df_config.load(
        str(model_dir / "config.ini"),
        config_must_exist=True,
        allow_defaults=True,
        allow_reload=True,
    )
    p = ModelParams()
    df_state = DF(
        sr=p.sr,
        fft_size=p.fft_size,
        hop_size=p.hop_size,
        nb_bands=p.nb_erb,
        min_nb_erb_freqs=p.min_nb_freqs,
    )
    model, epoch = df_ckpt_load_model(str(model_dir / "checkpoints"), df_state, epoch="best")
    model = model.eval()
    return model


# --------------------------------------------------------------------------
# Reference (offline, upstream, unmodified) forward for validation
# --------------------------------------------------------------------------
def offline_reference_enhance(model, waveform: torch.Tensor, variant: int):
    """Run upstream's own (batched, non-streaming) forward on a whole utterance."""
    from df.enhance import df_features

    p = ModelParams()
    df_state = DF(
        sr=p.sr,
        fft_size=p.fft_size,
        hop_size=p.hop_size,
        nb_bands=p.nb_erb,
        min_nb_erb_freqs=p.min_nb_freqs,
    )
    spec, erb_feat, spec_feat = df_features(waveform, df_state, p.nb_df, device="cpu")
    with torch.no_grad():
        enhanced_spec = model(spec.clone(), erb_feat, spec_feat)[0]
    # ISTFT back to waveform using df_state's own synthesis (Rust) for a fair
    # reference; but to keep this pure-python and directly comparable with
    # our own STFT, we do the synthesis ourselves with the same STFT module.
    return enhanced_spec


# --------------------------------------------------------------------------
# Export / verify / quantize
# --------------------------------------------------------------------------
def build_streaming_model(variant: int, model_dir: str = None):
    model = load_variant_model(variant, model_dir)
    cls = VARIANT_CLASSES[variant]
    stream = cls(model, variant).eval()
    return stream


def initial_inputs(stream: StreamingDeepFilterNet):
    hop = stream.hop_size
    state = stream.bank.pack(stream.initial_state())
    return (
        torch.zeros(1, hop, dtype=torch.float32),
        torch.zeros(1, hop, dtype=torch.float32),
        torch.zeros(1, hop, dtype=torch.float32),
        state,
    )


def add_metadata(output_path: str, stream: StreamingDeepFilterNet, model_type: str):
    model = onnx.load(output_path)
    metadata = {
        "model_type": model_type,
        "sample_rate": str(stream.sr),
        "frame_length": str(stream.n_fft),
        "frame_shift": str(stream.hop_size),
        "state_size": str(stream.state_size),
        "streaming": "1",
        # Total algorithmic delay in hops (conv_lookahead + df_lookahead):
        # output for hop t reflects audio from hop (t - total_delay), so the
        # C++ backend must drop the first `total_delay` outputs and flush
        # `total_delay` extra zero-hops at end-of-stream to stay sample-
        # count aligned with the input.
        "delay_hops": str(stream.total_delay),
    }
    del model.metadata_props[:]
    for key, value in metadata.items():
        item = model.metadata_props.add()
        item.key = key
        item.value = value
    onnx.checker.check_model(model)
    onnx.save(model, output_path)


def load_test_waveform(sample_rate: int, num_frames_hint: int, hop_size: int) -> torch.Tensor:
    import torchaudio

    wav_path = REPO_ROOT / "public" / "wavs" / "zh.wav"
    if wav_path.exists():
        wav, sr = torchaudio.load(str(wav_path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        min_len = num_frames_hint * hop_size
        if wav.shape[1] < min_len:
            reps = min_len // wav.shape[1] + 1
            wav = wav.repeat(1, reps)
        wav = wav[:, :min_len]
        return wav.to(torch.float32)
    torch.manual_seed(20260815)
    return (torch.rand(1, num_frames_hint * hop_size) * 2.0 - 1.0).to(torch.float32)


def run_streaming_pytorch(stream: StreamingDeepFilterNet, waveform: torch.Tensor):
    hop = stream.hop_size
    state = stream.initial_state()
    outputs = []
    with torch.no_grad():
        analysis_cache = torch.zeros(1, hop)
        synthesis_cache = torch.zeros(1, hop)
        state_flat = stream.bank.pack(state)
        for offset in range(0, waveform.shape[1] - hop + 1, hop):
            speech = waveform[:, offset : offset + hop]
            enhanced, analysis_cache, synthesis_cache, state_flat = stream(
                speech, analysis_cache, synthesis_cache, state_flat
            )
            outputs.append(enhanced)
    return torch.cat(outputs, dim=1)


def verify_vs_offline(stream: StreamingDeepFilterNet, waveform: torch.Tensor):
    from df.enhance import df_features

    p = ModelParams()
    df_state = DF(
        sr=p.sr,
        fft_size=p.fft_size,
        hop_size=p.hop_size,
        nb_bands=p.nb_erb,
        min_nb_erb_freqs=p.min_nb_freqs,
    )
    spec, erb_feat, spec_feat = df_features(waveform, df_state, p.nb_df, device="cpu")
    with torch.no_grad():
        enhanced_spec = stream.model(spec.clone(), erb_feat, spec_feat)[0]
    # Synthesize offline enhanced_spec with our own STFT synthesis for an
    # apples-to-apples comparison (avoids relying on rust ISTFT rounding).
    T = enhanced_spec.shape[2]
    synthesis_cache = torch.zeros(1, stream.hop_size)
    frames = []
    with torch.no_grad():
        for t in range(T):
            frame = enhanced_spec[:, :, t : t + 1]
            enhanced, synthesis_cache = stream.stft.synthesis(frame, synthesis_cache)
            frames.append(enhanced)
    offline_wave = torch.cat(frames, dim=1)

    streaming_wave = run_streaming_pytorch(stream, waveform)

    # The streaming wrapper lags the offline (whole-utterance, non-causal)
    # reference by `total_delay` hops: it needs that many extra hops of
    # future audio before it can produce the frame the offline model
    # already knows about immediately. So streaming_wave[t] corresponds to
    # offline_wave[t - total_delay*hop], i.e. the delay offset must be
    # applied to the STREAMING side, not the offline side.
    delay_samples = stream.total_delay * stream.hop_size
    n = min(offline_wave.shape[1], streaming_wave.shape[1] - delay_samples)
    n = max(n, 0)
    a = offline_wave[:, :n]
    b = streaming_wave[:, delay_samples : delay_samples + n]
    diff = float(torch.max(torch.abs(a - b))) if n > 0 else float("nan")
    return diff


def export_onnx(stream: StreamingDeepFilterNet, output_path: str):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy_inputs = initial_inputs(stream)
    torch.onnx.export(
        stream,
        dummy_inputs,
        str(output),
        input_names=["speech", "analysis_cache", "synthesis_cache", "state_in"],
        output_names=["enhanced", "analysis_cache_out", "synthesis_cache_out", "state_out"],
        opset_version=OPSET_VERSION,
        dynamo=False,
    )


def verify_onnx(stream: StreamingDeepFilterNet, output_path: str, waveform: torch.Tensor):
    hop = stream.hop_size
    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    torch_state = list(initial_inputs(stream)[1:])
    ort_state = [v.numpy().copy() for v in initial_inputs(stream)[1:]]
    wav_diff = 0.0
    state_diff = 0.0
    outputs = []
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
            for tc, oc in zip(torch_out[1:], ort_out[1:]):
                state_diff = max(state_diff, float(np.max(np.abs(tc.numpy() - oc))))
            torch_state = [v.detach().clone() for v in torch_out[1:]]
            ort_state = [v.copy() for v in ort_out[1:]]
    return wav_diff, state_diff


def quantize_onnx_model(input_path: str, output_path: str):
    model = onnx.load(input_path)
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
    )
    preprocess_inits = [
        init.name
        for init in model.graph.initializer
        if any(k in init.name.lower() for k in preprocess_keywords)
    ]
    for node in model.graph.node:
        node_name = node.name.lower()
        if any(inp in preprocess_inits for inp in node.input):
            nodes_to_exclude.append(node.name)
            continue
        if any(k in node_name for k in preprocess_keywords):
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


def run_one(variant: int, args):
    info = VARIANTS[variant]
    print(f"\n=== DeepFilterNet variant {variant} ===")
    stream = build_streaming_model(variant, args.model_dir)
    print(f"state_size = {stream.state_size}, total_delay(hops) = {stream.total_delay}")

    output_path = args.onnx_path or f"public/models/deepfilternet{info['suffix']}.onnx"
    output_path = str(REPO_ROOT / output_path) if not os.path.isabs(output_path) else output_path

    waveform = load_test_waveform(stream.sr, num_frames_hint=500, hop_size=stream.hop_size)

    offline_diff = float("nan")
    if args.verify:
        offline_diff = verify_vs_offline(stream, waveform)
        print(f"streaming-PyTorch vs offline-PyTorch max abs diff: {offline_diff:.8g}")
        if not (offline_diff < 1e-4):
            print(
                f"WARNING: variant {variant} offline diff {offline_diff:.8g} "
                "exceeds 1e-4 threshold"
            )

    export_onnx(stream, output_path)
    add_metadata(output_path, stream, info["model_type"])
    size = os.path.getsize(output_path)
    print(f"Exported: {output_path} ({size:,} bytes, {size/1024/1024:.2f} MB)")

    onnx_wav_diff = onnx_state_diff = float("nan")
    if args.verify:
        onnx_wav_diff, onnx_state_diff = verify_onnx(stream, output_path, waveform)
        print(
            f"streaming-PyTorch vs streaming-ONNX max abs diff: "
            f"waveform={onnx_wav_diff:.8g}, state={onnx_state_diff:.8g}"
        )

    qsize = None
    if args.quantize:
        quantized_path = output_path.replace(".onnx", ".int8.onnx")
        quantize_onnx_model(output_path, quantized_path)
        add_metadata(quantized_path, stream, info["model_type"])
        qsize = os.path.getsize(quantized_path)
        print(f"Int8 file size: {qsize:,} bytes ({qsize/1024/1024:.2f} MB)")

    return {
        "variant": variant,
        "state_size": stream.state_size,
        "offline_diff": offline_diff,
        "onnx_wav_diff": onnx_wav_diff,
        "onnx_state_diff": onnx_state_diff,
        "float_size": size,
        "int8_size": qsize,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Export streaming DeepFilterNet v1/v2/v3")
    parser.add_argument("--variant", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--onnx-path", default=None)
    parser.add_argument("--opset", type=int, default=OPSET_VERSION)
    parser.add_argument("--verify", type=int, default=1)
    parser.add_argument("--quantize", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    variants = [args.variant] if args.variant else [1, 2, 3]
    results = []
    for v in variants:
        results.append(run_one(v, args))

    print("\n=== Summary ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
