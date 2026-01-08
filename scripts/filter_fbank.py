#! /usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    """HTK Mel scale conversion"""
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    """HTK Mel scale inverse conversion"""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def round_up_to_nearest_power_of_two(n: int) -> int:
    """
    Round up to the nearest power of two.
    Ported from kaldi/src/base/kaldi-math.cc
    """
    assert n > 0
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


class Filterbank(nn.Module):
    """
    Kaldi-style Filterbank implementation in PyTorch.

    Args:
        sample_rate (int): Sampling rate of the waveform. Default: 16000.
        num_mel_bins (int): Number of Mel filter bins. Default: 80.
        frame_length (float): Length of each frame in milliseconds. Default: 25.0.
        frame_shift (float): Step between frames in milliseconds. Default: 10.0.
        preemph_coeff (float): Pre-emphasis coefficient. 0.0 to disable. Default: 0.97.
        dither (float): Amplitude of dither noise added to waveform. 0.0 to disable. Default: 1.0.
        window_fn (Callable): Window function (e.g., torch.hamming_window). Default: torch.hamming_window.
        remove_dc_offset (bool): If True, remove DC offset from each frame. Default: True.
        round_to_power_of_two (bool): If True, round n_fft to the nearest power of two. Default: True.
        snip_edges (bool): If True, only output complete frames. If False, include all frames by padding. Default: True.
        use_energy (bool): If True, append log-energy as the first feature. Default: False.
        raw_energy (bool): If True, calculate energy before windowing. If False, calculate after windowing. Default: True.
        log_energy_floor (float): Floor for log-energy. Default: 0.0 (effectively epsilon).
        low_freq (float): Lower frequency boundary for Mel filters. Default: 20.0.
        high_freq (float): Upper frequency boundary for Mel filters. 0 or None for Nyquist. Default: 0.0.
        eps (float): Small value to prevent log(0). Default: 1e-10.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        num_mel_bins: int = 80,
        frame_length: float = 25.0,
        frame_shift: float = 10.0,
        preemph_coeff: float = 0.97,
        dither: float = 1.0,
        window_fn=torch.hamming_window,
        remove_dc_offset: bool = True,
        round_to_power_of_two: bool = True,
        snip_edges: bool = True,
        use_energy: bool = False,
        raw_energy: bool = True,
        log_energy_floor: float = 0.0,
        low_freq: float = 20.0,
        high_freq: float = 0.0,
        eps: float = torch.finfo(torch.float32).eps,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.frame_length_ms = frame_length
        self.frame_shift_ms = frame_shift
        self.preemph_coeff = preemph_coeff
        self.dither = dither
        self.snip_edges = snip_edges
        self.use_energy = use_energy
        self.raw_energy = raw_energy
        self.log_energy_floor = log_energy_floor
        self.low_freq = low_freq
        self.high_freq = high_freq if high_freq > 0 else sample_rate / 2.0
        self.remove_dc_offset = remove_dc_offset
        self.round_to_power_of_two = round_to_power_of_two
        self.eps = eps

        # Convert time units from ms to samples
        self.win_length = int(round(frame_length * sample_rate / 1000))
        self.hop_length = int(round(frame_shift * sample_rate / 1000))

        # Determine n_fft (must be power of two and >= win_length if rounded)
        if self.round_to_power_of_two:
            self.n_fft = round_up_to_nearest_power_of_two(self.win_length)
        else:
            self.n_fft = self.win_length

        # Number of STFT bins (Kaldifeat style, removing Nyquist bin)
        self.n_stft = self.n_fft // 2

        # Create and register window (ONNX compatible)
        window = window_fn(self.win_length, periodic=False, dtype=torch.float32)
        self.register_buffer("window", window)

        # Create and register Mel filterbank
        mel_filters = self._create_mel_filterbank()
        self.register_buffer("mel_filters", mel_filters)  # [num_mel_bins, n_stft]

        # Pre-compute DFT matrix (for ONNX-compatible FFT)
        # DFT: X[k] = sum(x[n] * exp(-2πi*k*n/N))
        # For rfft, we only need k = 0, 1, ..., N/2
        dft_matrix_real, dft_matrix_imag = self._create_dft_matrix()
        self.register_buffer("dft_matrix_real", dft_matrix_real)  # [n_fft, n_stft]
        self.register_buffer("dft_matrix_imag", dft_matrix_imag)  # [n_fft, n_stft]

        # Register constants for ONNX compatibility
        self.register_buffer("_eps_tensor", torch.tensor(eps, dtype=torch.float32))
        if log_energy_floor > 0:
            self.register_buffer(
                "_log_energy_floor", torch.tensor(log_energy_floor, dtype=torch.float32)
            )

    def _create_mel_filterbank(self) -> torch.Tensor:
        """
        Creates the Mel filterbank matrix aligned with Kaldi/Kaldifeat.

        Returns:
            mel_filters: [num_mel_bins, n_stft] Mel filterbank matrix
        """
        num_fft_bins = self.n_stft

        # 1. Frequency boundaries
        low_freq = self.low_freq
        high_freq = self.high_freq

        # 2. FFT bin width
        fft_bin_width = self.sample_rate / self.n_fft

        # 3. Convert to Mel scale
        mel_low = _hz_to_mel(torch.tensor(low_freq, dtype=torch.float32))
        mel_high = _hz_to_mel(torch.tensor(high_freq, dtype=torch.float32))

        # 4. Mel frequency spacing (Kaldi uses num_bins + 1 for denominator)
        mel_freq_delta = (mel_high - mel_low) / (self.num_mel_bins + 1)

        # 5. Build filter matrix
        filters = torch.zeros(self.num_mel_bins, num_fft_bins, dtype=torch.float32)

        for bin_idx in range(self.num_mel_bins):
            # Left, Center, Right Mel frequencies
            left_mel = mel_low + bin_idx * mel_freq_delta
            center_mel = mel_low + (bin_idx + 1) * mel_freq_delta
            right_mel = mel_low + (bin_idx + 2) * mel_freq_delta

            for i in range(num_fft_bins):
                # Frequency of each FFT bin
                freq = fft_bin_width * i
                mel = _hz_to_mel(torch.tensor(freq, dtype=torch.float32))

                # Kaldi uses strict inequality: mel > left_mel && mel < right_mel
                if mel > left_mel and mel < right_mel:
                    if mel <= center_mel:
                        weight = (mel - left_mel) / (center_mel - left_mel)
                    else:
                        weight = (right_mel - mel) / (right_mel - center_mel)
                    filters[bin_idx, i] = weight.item()

        return filters  # [num_mel_bins, n_stft]

    def _create_dft_matrix(self) -> tuple:
        """
        Creates DFT matrix for ONNX-compatible FFT calculation.

        DFT: X[k] = sum(x[n] * exp(-2πi*k*n/N))

        For rfft, we only need k = 0, 1, ..., N/2
        Kaldifeat removes the last one (Nyquist), so we have N/2 points.

        Returns:
            dft_matrix_real: [n_fft, n_stft] - cos(2πkn/N)
            dft_matrix_imag: [n_fft, n_stft] - -sin(2πkn/N)
        """
        n = torch.arange(self.n_fft, dtype=torch.float32).unsqueeze(1)  # [n_fft, 1]
        k = torch.arange(self.n_stft, dtype=torch.float32).unsqueeze(0)  # [1, n_stft]

        # Angle: 2π * k * n / N
        angles = 2 * math.pi * k * n / self.n_fft  # [n_fft, n_stft]

        # Real and Imaginary parts of DFT matrix
        # exp(-iθ) = cos(θ) - i*sin(θ)
        dft_matrix_real = torch.cos(angles)  # [n_fft, n_stft]
        dft_matrix_imag = -torch.sin(angles)  # [n_fft, n_stft]

        return dft_matrix_real, dft_matrix_imag

    # ========================================
    # Modular processing functions
    # ========================================

    def extract_frames_unfold(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Extract frames using unfold.
        Note: unfold may fail ONNX export with dynamic input sizes.

        Args:
            waveforms: [B, num_samples]

        Returns:
            frames: [B, num_frames, win_length]
        """
        return waveforms.unfold(1, self.win_length, self.hop_length)

    def extract_frames_as_strided(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Extract frames using gather + indexing (ONNX compatible).
        This approach has better ONNX export compatibility than unfold.

        Args:
            waveforms: [B, num_samples]

        Returns:
            frames: [B, num_frames, win_length]
        """
        batch_size = waveforms.size(0)
        num_samples = waveforms.size(1)
        num_frames = (num_samples - self.win_length) // self.hop_length + 1

        # Build index matrix
        frame_indices = torch.arange(
            self.win_length, device=waveforms.device
        ).unsqueeze(
            0
        )  # [1, win_length]
        start_indices = (
            torch.arange(num_frames, device=waveforms.device).unsqueeze(1)
            * self.hop_length
        )  # [num_frames, 1]
        indices = frame_indices + start_indices  # [num_frames, win_length]

        # Expand to batch dimension
        indices = indices.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [B, num_frames, win_length]

        # Extract frames using gather
        frames = torch.gather(
            waveforms.unsqueeze(1).expand(-1, num_frames, -1), 2, indices
        )  # [B, num_frames, win_length]

        return frames

    def remove_dc_offset_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Remove DC offset from each frame.

        Args:
            frames: [B, num_frames, win_length]

        Returns:
            frames: [B, num_frames, win_length]
        """
        if self.remove_dc_offset:
            frame_means = frames.mean(dim=-1, keepdim=True)
            frames = frames - frame_means
        return frames

    def preemphasize(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Apply pre-emphasis (aligned with Kaldi).

        Kaldi frame-level pre-emphasis:
        for (i = frame_length - 1; i > 0; i--)
            data[i] -= preemph_coeff * data[i-1];
        data[0] -= preemph_coeff * data[0];

        Args:
            frames: [B, num_frames, win_length]

        Returns:
            preemph_frames: [B, num_frames, win_length]
        """
        if self.preemph_coeff <= 0:
            return frames

        preemph_frames = torch.zeros_like(frames)
        preemph_frames[..., 1:] = (
            frames[..., 1:] - self.preemph_coeff * frames[..., :-1]
        )
        preemph_frames[..., 0] = frames[..., 0] * (1.0 - self.preemph_coeff)
        return preemph_frames

    def compute_power_spectrum_fft(self, windowed: torch.Tensor) -> torch.Tensor:
        """
        Computes power spectrum using torch.fft.rfft (efficient but not ONNX compatible).

        Args:
            windowed: [B, num_frames, win_length]

        Returns:
            power_spec: [B, num_frames, n_stft]
        """
        # Zero-padding to n_fft length
        if self.n_fft > self.win_length:
            padding = self.n_fft - self.win_length
            windowed = F.pad(windowed, (0, padding))

        # FFT
        spec = torch.fft.rfft(windowed, n=self.n_fft)

        # Align with Kaldifeat: magnitude spectrum -> remove Nyquist bin -> power spectrum
        magnitude_spec = torch.abs(spec)  # [B, num_frames, n_fft//2+1]
        magnitude_spec = magnitude_spec[
            ..., :-1
        ]  # [B, num_frames, n_stft] remove Nyquist
        power_spec = magnitude_spec.pow(2)  # [B, num_frames, n_stft]

        return power_spec

    def compute_power_spectrum_dft(self, windowed: torch.Tensor) -> torch.Tensor:
        """
        Computes power spectrum using DFT matrix multiplication (ONNX compatible).

        DFT: X[k] = sum(x[n] * exp(-2πi*k*n/N))
        |X[k]|^2 = real^2 + imag^2

        Args:
            windowed: [B, num_frames, win_length]

        Returns:
            power_spec: [B, num_frames, n_stft]
        """
        # Zero-padding to n_fft length
        if self.n_fft > self.win_length:
            padding = self.n_fft - self.win_length
            windowed = F.pad(windowed, (0, padding))

        # DFT via matrix multiplication
        # Real: sum(x[n] * cos(2πkn/N))
        # Imag: sum(x[n] * (-sin(2πkn/N)))
        spec_real = torch.matmul(
            windowed, self.dft_matrix_real
        )  # [B, num_frames, n_stft]
        spec_imag = torch.matmul(
            windowed, self.dft_matrix_imag
        )  # [B, num_frames, n_stft]

        # Power Spectrum: |X|^2 = real^2 + imag^2
        power_spec = spec_real.pow(2) + spec_imag.pow(2)  # [B, num_frames, n_stft]

        return power_spec

    def apply_mel_filterbank(self, power_spec: torch.Tensor) -> torch.Tensor:
        """
        Applies Mel filterbank and computes log.

        Args:
            power_spec: [B, num_frames, n_stft]

        Returns:
            log_mel_spec: [B, num_frames, num_mel_bins]
        """
        mel_spec = torch.matmul(power_spec, self.mel_filters.t())
        log_mel_spec = torch.log(torch.clamp(mel_spec, min=self.eps))
        return log_mel_spec

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Computes FBank features for the input waveforms.
        Uses a modular processing flow aligned with Kaldi/Kaldifeat.

        Args:
            waveforms: Input audio waveforms of shape [batch_size, num_samples]

        Returns:
            features: FBank features
                - If use_energy=False: [batch_size, num_frames, num_mel_bins]
                - If use_energy=True: [batch_size, num_frames, num_mel_bins + 1]

        Note:
            Automatically detects ONNX export environment:
            - During ONNX export: uses DFT matrix and gather (fully compatible).
            - During normal execution: uses torch.fft.rfft and unfold (efficient).
        """
        if waveforms.dim() != 2:
            raise ValueError(
                f"Expected 2D tensor (batch, samples), got {waveforms.shape}"
            )

        # 1. Dither (add random noise during training)
        if self.dither > 0:
            waveforms = waveforms + torch.randn_like(waveforms) * self.dither

        # 2. Handle snip_edges mode
        if not self.snip_edges:
            pad_amount = self.win_length // 2
            waveforms = F.pad(waveforms, (pad_amount, pad_amount), mode="reflect")

        # 3. Framing (gather for ONNX, unfold otherwise)
        if torch.onnx.is_in_onnx_export():
            frames = self.extract_frames_as_strided(waveforms)
        else:
            frames = self.extract_frames_unfold(waveforms)

        # 4. DC offset removal
        frames = self.remove_dc_offset_frames(frames)

        # 5. Compute raw energy (before pre-emphasis, if requested)
        if self.use_energy and self.raw_energy:
            raw_energy = torch.sum(frames**2, dim=-1)
            raw_energy = torch.clamp(raw_energy, min=self.eps)
            log_energy = torch.log(raw_energy)
            if self.log_energy_floor > 0:
                log_energy = torch.maximum(log_energy, self._log_energy_floor)

        # 6. Pre-emphasis
        frames = self.preemphasize(frames)

        # 7. Windowing
        windowed = frames * self.window

        # 8. Power Spectrum (DFT matrix for ONNX, FFT otherwise)
        if torch.onnx.is_in_onnx_export():
            power_spec = self.compute_power_spectrum_dft(windowed)
        else:
            power_spec = self.compute_power_spectrum_fft(windowed)

        # 9. Compute windowed energy (if requested)
        if self.use_energy and not self.raw_energy:
            log_energy = torch.log(torch.sum(power_spec, dim=-1).clamp(min=self.eps))
            if self.log_energy_floor > 0:
                log_energy = torch.maximum(log_energy, self._log_energy_floor)

        # 10. Mel Filterbank + Log
        log_mel_spec = self.apply_mel_filterbank(power_spec)

        # 11. Concat energy (as the first dimension)
        if self.use_energy:
            features = torch.cat([log_energy.unsqueeze(-1), log_mel_spec], dim=-1)
        else:
            features = log_mel_spec

        return features
