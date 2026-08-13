// Pitch (F0) estimator for TEN VAD.
//
// This is a port of TEN VAD's src/pitch_est.cc, which is itself a modified
// version of the pitch estimation code from Xiph.Org / Mozilla's LPCNet
// (BSD-2-Clause / BSD-3-Clause). See the TEN VAD repo's NOTICES file.
//
//   Copyright (c) 2025 Agora (TEN Framework), Apache-2.0
//   Copyright (c) 2017-2019 Mozilla, Xiph.Org Foundation
//
// Why this lives in C++ rather than in the ONNX graph
// ---------------------------------------------------
// Every other frontend in this project is traced into its model's ONNX graph.
// This one cannot be: the algorithm is a Levinson-Durbin LPC solve, an IIR
// decimation filter, a Viterbi-style max-path search with backtracking, and a
// weighted linear regression, all carrying state across frames. That is
// data-dependent control flow, not a fixed tensor pipeline.
//
// So TEN VAD's ONNX graph takes pitch as an input, and this class produces it.
// The 40 log-mel features are still computed inside the graph.

#pragma once

#include <cstddef>
#include <vector>

namespace VadFilterOnnx {

class PitchEstimator {
  public:
    // sample_rate must be 16000; hop_size is the frame advance in samples
    // (256 = 16 ms, matching TenVadModel's frame shift).
    explicit PitchEstimator(int sample_rate = 16000, int hop_size = 256);

    // Clear all cross-frame state. Call at stream start.
    void reset();

    // Consume one analysis window and return the current F0 estimate.
    //
    // `data` is the same [frame_length] window handed to the ONNX model, in
    // normalized float. Only the newest `hop_size` samples advance the
    // estimator's internal state; the rest supplies the spectrum used for the
    // LPC fit. `n` must be at least hop_size.
    //
    // Returns F0 in Hz, or 0.0f when the frame is unvoiced. This matches the
    // convention the model's normalization statistics were fitted with, so the
    // value can be fed to the graph's `pitch` input as-is.
    float process(const float *data, int n);

    // Whether the most recent frame was classified as voiced.
    bool voiced() const { return voiced_; }

    // Correlation threshold above which a frame counts as voiced. Upstream
    // uses 0.4 at a 4 kHz internal rate.
    void set_voiced_threshold(float threshold) { voiced_threshold_ = threshold; }

  private:
    // Power spectrum of the (pre-emphasized, windowed, zero-padded) frame.
    void compute_power_spectrum(const float *data, int n);
    // 18 flooded log-band energies from the power spectrum.
    void compute_band_energy();
    // DCT-II / DCT-III over the 18 bands.
    void dct(const float *in, float *out) const;
    void idct(const float *in, float *out) const;
    // Fit an order-16 LPC to the band energies; returns the residual error.
    float lpc_from_bands(const float *bands);
    // Levinson-Durbin.
    float celt_lpc(const float *ac, int p, float *lpc) const;
    // LPC inverse filter + pitch prefilter over one hop.
    void lpc_filter(const float *hop);
    // 5-section biquad IIR lowpass, then decimate by procResampleRate.
    void decimate_into_exc_buffer();
    // Normalized cross-correlation for the two half-hop sub-frames.
    void update_cross_correlation();
    // Viterbi max-path search plus backtracking and regression.
    float estimate_period();

    // Real-input FFT helper (radix-2, in-place on interleaved complex).
    void real_fft(std::vector<float> *buffer, int n, bool inverse) const;

    int sample_rate_;
    int hop_size_;
    int fft_size_ = 1024;
    int n_bins_ = 513;
    int window_size_ = 768;

    int proc_fs_ = 4000;
    int resample_rate_ = 4; // 16000 / 4000
    int min_period_ = 8;    // 32 / 4  -> 500 Hz
    int max_period_ = 64;   // 256 / 4 -> 62.5 Hz
    int diff_period_ = 56;
    int n_feat_ = 0;      // correlation frames retained (<= 12)
    int half_hop_ = 0;    // hop_size / (resample_rate * 2)
    int exc_buf_len_ = 0; // max_period + ceil(hop/resample) + 1
    int input_q_len_ = 0; // max(80, hop) + hop

    float voiced_threshold_ = 0.4f;
    bool voiced_ = false;
    float pitch_hz_ = 0.0f;

    std::vector<float> window_;    // 768-point Hann
    std::vector<float> dct_table_; // 18 x 18
    std::vector<float> fft_buf_;
    std::vector<float> power_;  // [n_bins_]
    std::vector<float> bands_;  // [18]
    std::vector<float> cepst_;  // [18]
    std::vector<float> lpc_;    // [16]
    std::vector<float> interp_; // [n_bins_]

    std::vector<float> input_q_;
    std::vector<float> aligned_in_;
    std::vector<float> lpc_out_;
    std::vector<float> pitch_mem_;
    float pitch_filt_ = 0.0f;
    float last_sample_ = 0.0f;

    // Biquad decimation state: [section][2].
    std::vector<float> biquad_state_;
    std::vector<float> resample_buf_;

    std::vector<float> exc_buf_;
    std::vector<float> exc_buf_sq_;

    // Cross-correlation ring buffer: [n_feat_ * 2][max_period_ + 1].
    std::vector<std::vector<float>> xcorr_;
    std::vector<std::vector<float>> xcorr_tmp_;
    std::vector<float> xcorr_inst_;
    std::vector<float> frame_weight_;
    std::vector<float> frame_weight_norm_;
    int xcorr_offset_ = 0;

    // Max-path tracking.
    std::vector<float> max_path_[2];
    std::vector<std::vector<int>> pitch_prev_;
    float max_path_all_ = 0.0f;
    int best_period_ = 0;
};

} // namespace VadFilterOnnx
