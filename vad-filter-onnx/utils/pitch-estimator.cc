// Port of TEN VAD's src/pitch_est.cc (LPCNet-derived). See pitch-estimator.h
// for attribution and for why this is not part of the ONNX graph.

#include "utils/pitch-estimator.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace VadFilterOnnx {

namespace {

constexpr int kNumBands = 18;
constexpr int kLpcOrder = 16;
constexpr int kXcorrTrainingOffset = 80;
constexpr int kMinPeriod16k = 32;
constexpr int kMaxPeriod16k = 256;
constexpr int kFeatTimeWindowMs = 40;
constexpr int kFeatMaxFrames = 12;
constexpr int kBiquadSections = 5;
constexpr float kMaxPathWeight = 0.02f;
constexpr float kPcmScale = 32768.0f;
constexpr float kPreemph = 0.97f;

// M_PI is not standard C++ and MSVC does not define it without
// _USE_MATH_DEFINES, so carry our own.
constexpr float kPi = 3.14159265358979323846f;

// The band layout is expressed against an 80-point reference FFT and scaled to
// the actual FFT size at runtime.
constexpr int kAssumedFftForBands = 80;
constexpr int kBandStart[kNumBands] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,
                                        10, 12, 14, 16, 20, 24, 28, 34, 40 };
constexpr float kBandLpcComp[kNumBands] = { 0.8f,      1.0f,  1.0f,  1.0f,  1.0f,      1.0f,
                                           1.0f,      1.0f,  0.666667f, 0.5f,  0.5f,  0.5f,
                                           0.333333f, 0.25f, 0.25f, 0.2f,  0.166667f, 0.173913f };

// 16 kHz -> 4 kHz decimation lowpass, 5 biquad sections.
constexpr float kBiquadB[kBiquadSections][3] = { { 1.0f, 1.198825e+00f, 1.0f },
                                                { 1.0f, -5.674614e-01f, 1.0f },
                                                { 1.0f, -1.099061e+00f, 1.0f },
                                                { 1.0f, -1.265846e+00f, 1.0f },
                                                { 1.0f, -1.318849e+00f, 1.0f } };
constexpr float kBiquadA[kBiquadSections][3] = { { 1.0f, -1.445267e+00f, 5.463974e-01f },
                                                { 1.0f, -1.426720e+00f, 6.820138e-01f },
                                                { 1.0f, -1.408255e+00f, 8.286664e-01f },
                                                { 1.0f, -1.400909e+00f, 9.240320e-01f },
                                                { 1.0f, -1.408242e+00f, 9.789776e-01f } };
constexpr float kBiquadG[kBiquadSections] = { 2.692541e-01f, 2.692541e-01f, 2.692541e-01f,
                                             2.692541e-01f, 2.692541e-01f };

} // namespace

PitchEstimator::PitchEstimator(int sample_rate, int hop_size)
    : sample_rate_(sample_rate), hop_size_(hop_size) {
    resample_rate_ = sample_rate_ / proc_fs_;
    min_period_ = kMinPeriod16k / resample_rate_;
    max_period_ = kMaxPeriod16k / resample_rate_;
    diff_period_ = max_period_ - min_period_;
    half_hop_ = hop_size_ / (resample_rate_ * 2);
    input_q_len_ = std::max(kXcorrTrainingOffset, hop_size_) + hop_size_;

    const int shift = static_cast<int>(
        std::ceil(static_cast<float>(hop_size_) / static_cast<float>(resample_rate_)));
    exc_buf_len_ = max_period_ + shift + 1;

    n_feat_ = static_cast<int>(std::ceil(kFeatTimeWindowMs * static_cast<float>(sample_rate_) /
                                        (static_cast<float>(hop_size_) * 1000.0f)));
    n_feat_ = std::min(n_feat_, kFeatMaxFrames);

    // 768-point periodic Hann, matching the model's analysis window.
    window_.resize(window_size_);
    for (int i = 0; i < window_size_; ++i) {
        const float phase = 2.0f * kPi * i / window_size_;
        window_[i] = 0.5f - 0.5f * std::cos(phase);
    }

    dct_table_.resize(kNumBands * kNumBands);
    for (int i = 0; i < kNumBands; ++i) {
        for (int j = 0; j < kNumBands; ++j) {
            float v = std::cos((i + 0.5f) * j * kPi / kNumBands);
            if (j == 0) {
                v *= std::sqrt(0.5f);
            }
            dct_table_[i * kNumBands + j] = v;
        }
    }

    power_.resize(n_bins_);
    bands_.resize(kNumBands);
    cepst_.resize(kNumBands);
    lpc_.resize(kLpcOrder);
    interp_.resize(n_bins_);
    fft_buf_.resize(2 * fft_size_);

    input_q_.resize(input_q_len_);
    aligned_in_.resize(hop_size_);
    lpc_out_.resize(hop_size_);
    pitch_mem_.resize(kLpcOrder);
    biquad_state_.resize(kBiquadSections * 2);
    resample_buf_.resize(hop_size_ * 2);
    exc_buf_.resize(exc_buf_len_);
    exc_buf_sq_.resize(exc_buf_len_);

    const int rows = n_feat_ * 2;
    xcorr_.assign(rows, std::vector<float>(max_period_ + 1, 0.0f));
    xcorr_tmp_.assign(rows, std::vector<float>(max_period_ + 1, 0.0f));
    xcorr_inst_.resize(max_period_ + 1);
    frame_weight_.resize(rows);
    frame_weight_norm_.resize(rows);
    pitch_prev_.assign(rows, std::vector<int>(max_period_, 0));
    max_path_[0].resize(max_period_);
    max_path_[1].resize(max_period_);

    reset();
}

void PitchEstimator::reset() {
    std::fill(input_q_.begin(), input_q_.end(), 0.0f);
    std::fill(aligned_in_.begin(), aligned_in_.end(), 0.0f);
    std::fill(lpc_out_.begin(), lpc_out_.end(), 0.0f);
    std::fill(pitch_mem_.begin(), pitch_mem_.end(), 0.0f);
    std::fill(lpc_.begin(), lpc_.end(), 0.0f);
    std::fill(biquad_state_.begin(), biquad_state_.end(), 0.0f);
    std::fill(resample_buf_.begin(), resample_buf_.end(), 0.0f);
    std::fill(exc_buf_.begin(), exc_buf_.end(), 0.0f);
    std::fill(exc_buf_sq_.begin(), exc_buf_sq_.end(), 0.0f);
    std::fill(frame_weight_.begin(), frame_weight_.end(), 0.0f);
    std::fill(frame_weight_norm_.begin(), frame_weight_norm_.end(), 0.0f);
    for (auto &row : xcorr_) {
        std::fill(row.begin(), row.end(), 0.0f);
    }
    for (auto &row : xcorr_tmp_) {
        std::fill(row.begin(), row.end(), 0.0f);
    }
    for (auto &row : pitch_prev_) {
        std::fill(row.begin(), row.end(), 0);
    }
    std::fill(max_path_[0].begin(), max_path_[0].end(), 0.0f);
    std::fill(max_path_[1].begin(), max_path_[1].end(), 0.0f);

    pitch_filt_ = 0.0f;
    last_sample_ = 0.0f;
    xcorr_offset_ = 0;
    max_path_all_ = 0.0f;
    best_period_ = 0;
    voiced_ = false;
    pitch_hz_ = 0.0f;
}

// Iterative radix-2 complex FFT over an interleaved [re, im, ...] buffer.
void PitchEstimator::real_fft(std::vector<float> *buffer, int n, bool inverse) const {
    float *data = buffer->data();

    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(data[2 * i], data[2 * j]);
            std::swap(data[2 * i + 1], data[2 * j + 1]);
        }
    }

    for (int len = 2; len <= n; len <<= 1) {
        const float ang = 2.0f * kPi / len * (inverse ? 1.0f : -1.0f);
        const float wr = std::cos(ang);
        const float wi = std::sin(ang);
        for (int i = 0; i < n; i += len) {
            float cwr = 1.0f;
            float cwi = 0.0f;
            for (int k = 0; k < len / 2; ++k) {
                const int a = 2 * (i + k);
                const int b = 2 * (i + k + len / 2);
                const float ur = data[a];
                const float ui = data[a + 1];
                const float vr = data[b] * cwr - data[b + 1] * cwi;
                const float vi = data[b] * cwi + data[b + 1] * cwr;
                data[a] = ur + vr;
                data[a + 1] = ui + vi;
                data[b] = ur - vr;
                data[b + 1] = ui - vi;
                const float nwr = cwr * wr - cwi * wi;
                cwi = cwr * wi + cwi * wr;
                cwr = nwr;
            }
        }
    }
}

void PitchEstimator::compute_power_spectrum(const float *data, int n) {
    // Upstream feeds the pitch module the pre-emphasized, Hann-windowed,
    // zero-padded frame's power spectrum (the same one the mel path uses), at
    // PCM scale.
    std::fill(fft_buf_.begin(), fft_buf_.end(), 0.0f);
    const int count = std::min(n, window_size_);
    const int offset = n - count; // use the newest samples if given more
    float prev = last_sample_;
    for (int i = 0; i < count; ++i) {
        const float sample = data[offset + i] * kPcmScale;
        fft_buf_[2 * i] = (sample - kPreemph * prev) * window_[i];
        prev = sample;
    }

    real_fft(&fft_buf_, fft_size_, false);
    for (int i = 0; i < n_bins_; ++i) {
        const float re = fft_buf_[2 * i];
        const float im = fft_buf_[2 * i + 1];
        power_[i] = re * re + im * im;
    }
}

void PitchEstimator::compute_band_energy() {
    const float rate = static_cast<float>(fft_size_) / kAssumedFftForBands;
    std::fill(bands_.begin(), bands_.end(), 0.0f);

    for (int i = 0; i < kNumBands - 1; ++i) {
        const int band_sz = static_cast<int>(std::round((kBandStart[i + 1] - kBandStart[i]) * rate));
        const int offset = static_cast<int>(std::round(kBandStart[i] * rate));
        for (int j = 0; j < band_sz; ++j) {
            const float frac = static_cast<float>(j) / band_sz;
            const int idx = std::min(n_bins_ - 1, offset + j);
            bands_[i] += (1.0f - frac) * power_[idx];
            bands_[i + 1] += frac * power_[idx];
        }
    }
    bands_[0] *= 2.0f;
    bands_[kNumBands - 1] *= 2.0f;
}

void PitchEstimator::dct(const float *in, float *out) const {
    const float ratio = std::sqrt(2.0f / kNumBands);
    for (int i = 0; i < kNumBands; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < kNumBands; ++j) {
            sum += in[j] * dct_table_[j * kNumBands + i];
        }
        out[i] = sum * ratio;
    }
}

void PitchEstimator::idct(const float *in, float *out) const {
    const float ratio = std::sqrt(2.0f / kNumBands);
    for (int i = 0; i < kNumBands; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < kNumBands; ++j) {
            sum += in[j] * dct_table_[i * kNumBands + j];
        }
        out[i] = sum * ratio;
    }
}

float PitchEstimator::celt_lpc(const float *ac, int p, float *lpc) const {
    std::fill(lpc, lpc + p, 0.0f);
    float error = ac[0];
    if (ac[0] == 0.0f) {
        return error;
    }

    for (int i = 0; i < p; ++i) {
        float rr = 0.0f;
        for (int j = 0; j < i; ++j) {
            rr += lpc[j] * ac[i - j];
        }
        rr += ac[i + 1];
        const float r = -rr / error;
        lpc[i] = r;
        for (int j = 0; j < ((i + 1) >> 1); ++j) {
            const float t1 = lpc[j];
            const float t2 = lpc[i - 1 - j];
            lpc[j] = t1 + r * t2;
            lpc[i - 1 - j] = t2 + r * t1;
        }
        error -= r * r * error;
        // Bail out once we have 30 dB of gain.
        if (error < 0.001f * ac[0]) {
            break;
        }
    }
    return error;
}

float PitchEstimator::lpc_from_bands(const float *bands) {
    // Interpolate band gains back onto the bin grid.
    const float rate = static_cast<float>(fft_size_) / kAssumedFftForBands;
    std::fill(interp_.begin(), interp_.end(), 0.0f);
    for (int i = 0; i < kNumBands - 1; ++i) {
        const int band_sz = static_cast<int>(std::round((kBandStart[i + 1] - kBandStart[i]) * rate));
        const int offset = static_cast<int>(std::round(kBandStart[i] * rate));
        for (int j = 0; j < band_sz; ++j) {
            const float frac = static_cast<float>(j) / band_sz;
            const int idx = std::min(n_bins_ - 1, offset + j);
            interp_[idx] = (1.0f - frac) * bands[i] + frac * bands[i + 1];
        }
    }
    interp_[n_bins_ - 1] = 0.0f; // drop Nyquist

    // Inverse transform the (real, symmetric) spectrum to get autocorrelation.
    std::fill(fft_buf_.begin(), fft_buf_.end(), 0.0f);
    for (int i = 0; i < n_bins_; ++i) {
        fft_buf_[2 * i] = interp_[i];
    }
    for (int i = 1; i < n_bins_ - 1; ++i) {
        fft_buf_[2 * (fft_size_ - i)] = interp_[i];
    }
    real_fft(&fft_buf_, fft_size_, true);

    float ac[kLpcOrder + 1];
    for (int i = 0; i < kLpcOrder + 1; ++i) {
        // Upstream's IFFT rescale is a flat 0.5 factor.
        ac[i] = fft_buf_[2 * i] * 0.5f;
    }

    // -40 dB noise floor, then lag windowing.
    const float dc_bias = window_size_ / 12.0f / 38.0f;
    ac[0] += ac[0] * 1e-4f + dc_bias;
    for (int i = 1; i < kLpcOrder + 1; ++i) {
        ac[i] *= (1.0f - 6e-5f * i * i);
    }

    return celt_lpc(ac, kLpcOrder, lpc_.data());
}

void PitchEstimator::lpc_filter(const float *hop) {
    // Slide the input queue and append this hop.
    std::memmove(input_q_.data(), input_q_.data() + hop_size_,
                 sizeof(float) * (input_q_len_ - hop_size_));
    std::memcpy(input_q_.data() + (input_q_len_ - hop_size_), hop, sizeof(float) * hop_size_);

    // Align the correlation window against the LPC training offset.
    const int offset = std::max(0, input_q_len_ - hop_size_ - kXcorrTrainingOffset);
    std::memcpy(aligned_in_.data(), input_q_.data() + offset, sizeof(float) * hop_size_);

    for (int i = 0; i < hop_size_; ++i) {
        float sum = aligned_in_[i];
        for (int j = 0; j < kLpcOrder; ++j) {
            sum += lpc_[j] * pitch_mem_[j];
        }
        std::memmove(pitch_mem_.data() + 1, pitch_mem_.data(), sizeof(float) * (kLpcOrder - 1));
        pitch_mem_[0] = aligned_in_[i];

        lpc_out_[i] = sum + 0.7f * pitch_filt_;
        pitch_filt_ = sum;
    }
}

void PitchEstimator::decimate_into_exc_buffer() {
    // 5 cascaded biquad sections, then keep every resample_rate_-th sample.
    for (int i = 0; i < hop_size_; ++i) {
        float x = lpc_out_[i];
        for (int s = 0; s < kBiquadSections; ++s) {
            float *w = &biquad_state_[s * 2];
            const float t = x - kBiquadA[s][1] * w[0] - kBiquadA[s][2] * w[1];
            x = kBiquadG[s] * (kBiquadB[s][0] * t + kBiquadB[s][1] * w[0] + kBiquadB[s][2] * w[1]);
            w[1] = w[0];
            w[0] = t;
        }
        resample_buf_[i] = x;
    }

    int count = 0;
    for (int i = 0; i < hop_size_; i += resample_rate_) {
        resample_buf_[count++] = resample_buf_[i];
    }

    std::memmove(exc_buf_.data(), exc_buf_.data() + count,
                 sizeof(float) * (exc_buf_len_ - count));
    std::memcpy(exc_buf_.data() + (exc_buf_len_ - count), resample_buf_.data(),
                sizeof(float) * count);

    for (int i = 0; i < exc_buf_len_; ++i) {
        exc_buf_sq_[i] = exc_buf_[i] * exc_buf_[i];
    }
}

void PitchEstimator::update_cross_correlation() {
    // Shift the per-frame energy weights to make room for this frame.
    for (int i = 0; i < n_feat_ - 1; ++i) {
        frame_weight_[2 * i] = frame_weight_[2 * (i + 1)];
        frame_weight_[2 * i + 1] = frame_weight_[2 * (i + 1) + 1];
    }

    for (int sub = 0; sub < 2; ++sub) {
        const int acc = 2 * xcorr_offset_ + sub;
        const int offset = sub * half_hop_;
        const float *ref = exc_buf_.data() + (max_period_ + offset);
        const float *mv = exc_buf_.data() + offset;

        for (int lag = 0; lag < max_period_; ++lag) {
            float sum = 0.0f;
            for (int k = 0; k < half_hop_; ++k) {
                sum += ref[k] * mv[lag + k];
            }
            xcorr_inst_[lag] = sum;
        }

        float energy0 = 0.0f;
        for (int k = 0; k < half_hop_; ++k) {
            energy0 += exc_buf_sq_[max_period_ + offset + k];
        }
        frame_weight_[2 * (n_feat_ - 1) + sub] = energy0;

        float win_sum = 0.0f;
        for (int k = 0; k < half_hop_; ++k) {
            win_sum += exc_buf_sq_[offset + k];
        }

        float denom = std::max(1e-12f, win_sum + (1.0f + energy0));
        xcorr_[acc][0] = 2.0f * xcorr_inst_[0] / denom;

        for (int lag = 1; lag < max_period_; ++lag) {
            win_sum = std::max(0.0f, win_sum - exc_buf_sq_[offset + lag - 1]);
            win_sum += exc_buf_sq_[offset + lag + half_hop_ - 1];
            denom = std::max(1e-12f, win_sum + (1.0f + energy0));
            xcorr_[acc][lag] = 2.0f * xcorr_inst_[lag] / denom;
        }

        // Suppress lags that look like octave errors of a stronger peak.
        for (int lag = 0; lag < max_period_ - 2 * min_period_; ++lag) {
            float peak = xcorr_[acc][(max_period_ + lag) / 2];
            peak = std::max(peak, xcorr_[acc][(max_period_ + lag + 2) / 2]);
            peak = std::max(peak, xcorr_[acc][(max_period_ + lag - 1) / 2]);
            if (xcorr_[acc][lag] < peak * 1.1f) {
                xcorr_[acc][lag] *= 0.8f;
            }
        }
    }

    if (++xcorr_offset_ >= n_feat_) {
        xcorr_offset_ = 0;
    }
}

float PitchEstimator::estimate_period() {
    const int rows = n_feat_ * 2;

    float total = 1e-15f;
    for (int i = 0; i < rows; ++i) {
        total += frame_weight_[i];
    }
    for (int i = 0; i < rows; ++i) {
        frame_weight_norm_[i] = frame_weight_[i] * (rows / total);
    }

    // Work on a copy so this frame's search does not disturb the ring buffer.
    for (int i = 0; i < rows; ++i) {
        xcorr_tmp_[i] = xcorr_[i];
    }

    // Slide the backtrack table.
    for (int sub = 0; sub < rows - 2; sub += 2) {
        pitch_prev_[sub] = pitch_prev_[sub + 2];
        pitch_prev_[sub + 1] = pitch_prev_[sub + 3];
    }

    for (int sub = rows - 2; sub < rows; ++sub) {
        int xc = sub + xcorr_offset_ * 2;
        if (xc >= rows) {
            xc -= rows;
        }

        for (int i = 0; i < diff_period_; ++i) {
            float best = max_path_all_ - 1e10f;
            pitch_prev_[sub][i] = best_period_;

            const int start = std::min(0, 4 - i);
            for (int j = start; j <= 4 && (i + j) < diff_period_; ++j) {
                const float cand = max_path_[0][i + j] - kMaxPathWeight * std::abs(j) * std::abs(j);
                if (cand > best) {
                    best = cand;
                    pitch_prev_[sub][i] = i + j;
                }
            }
            max_path_[1][i] = best + frame_weight_norm_[sub] * xcorr_tmp_[xc][i];
        }

        float best_path = -1e15f;
        int best_idx = 0;
        for (int i = 0; i < diff_period_; ++i) {
            if (max_path_[1][i] > best_path) {
                best_path = max_path_[1][i];
                best_idx = i;
            }
        }
        max_path_all_ = best_path;
        best_period_ = best_idx;

        max_path_[0] = max_path_[1];
        for (int i = 0; i < diff_period_; ++i) {
            max_path_[0][i] -= best_path;
        }
    }

    // Backward pass: recover the per-sub-frame period and the mean correlation.
    std::vector<int> periods(rows, 0);
    int cur = best_period_;
    float corr = 0.0f;
    for (int sub = rows - 1; sub >= 0; --sub) {
        periods[sub] = max_period_ - cur;
        int xc = sub + xcorr_offset_ * 2;
        if (xc >= rows) {
            xc -= rows;
        }
        corr += frame_weight_norm_[sub] * xcorr_tmp_[xc][cur];
        cur = pitch_prev_[sub][cur];
    }
    corr = std::max(0.0f, corr / static_cast<float>(rows));
    voiced_ = corr >= voiced_threshold_;

    // Weighted linear regression over the period contour.
    float sw = 0.0f, sx = 0.0f, sxx = 0.0f, sxy = 0.0f, sy = 0.0f;
    for (int sub = 0; sub < rows; ++sub) {
        const float w = frame_weight_norm_[sub];
        sw += w;
        sx += w * sub;
        sxx += w * sub * sub;
        sxy += w * sub * periods[sub];
        sy += w * periods[sub];
    }

    const float denom = sw * sxx - sx * sx;
    float slope = (sw * sxy - sx * sy) / (denom == 0.0f ? 1e-15f : denom);
    if (voiced_) {
        const float limit = (sy / sw) / (4.0f * 2.0f * n_feat_);
        slope = std::min(limit, std::max(-limit, slope));
    } else {
        slope = 0.0f;
    }
    const float intercept = (sy - slope * sx) / sw;
    return intercept + 5.5f * slope;
}

float PitchEstimator::process(const float *data, int n) {
    if (data == nullptr || n < hop_size_) {
        return 0.0f;
    }

    compute_power_spectrum(data, n);
    compute_band_energy();

    // Flooded log-band energies, then cepstrum.
    float log_max = -2.0f;
    float follow = -2.0f;
    std::vector<float> ly(kNumBands);
    for (int i = 0; i < kNumBands; ++i) {
        float v = std::log10(1e-2f + bands_[i]);
        v = std::max(log_max - 8.0f, std::max(follow - 2.5f, v));
        log_max = std::max(log_max, v);
        follow = std::max(follow - 2.5f, v);
        ly[i] = v;
    }
    dct(ly.data(), cepst_.data());

    // Cepstrum -> band gains -> LPC.
    float ex[kNumBands];
    idct(cepst_.data(), ex);
    for (int i = 0; i < kNumBands; ++i) {
        ex[i] = std::pow(10.0f, ex[i]) * kBandLpcComp[i];
    }
    lpc_from_bands(ex);

    // The estimator advances on the newest hop of raw (non-pre-emphasized)
    // samples; upstream passes inputTimeFIFO here, not the emphasized signal.
    std::vector<float> hop(hop_size_);
    const int hop_offset = n - hop_size_;
    for (int i = 0; i < hop_size_; ++i) {
        hop[i] = data[hop_offset + i] * kPcmScale;
    }
    last_sample_ = data[n - 1] * kPcmScale;

    lpc_filter(hop.data());
    decimate_into_exc_buffer();
    update_cross_correlation();

    const float period = estimate_period();
    pitch_hz_ = voiced_ ? static_cast<float>(proc_fs_) / std::max(1.0f, period) : 0.0f;
    return pitch_hz_;
}

} // namespace VadFilterOnnx
