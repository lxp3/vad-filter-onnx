#include "vad/segmenter.h"
#include <algorithm>
#include <stdexcept>

namespace VadFilterOnnx {

namespace {
int ceil_div(int a, int b) { return (a + b - 1) / b; }
} // namespace

Segmenter::Segmenter(const VadConfig &config, int frame_shift_samples, float threshold)
    : config_(config),
      frame_shift_(frame_shift_samples),
      threshold_(threshold),
      samples_per_ms_(config.sample_rate / 1000),
      speech_window_frames_(0),
      speech_threshold_frames_(0),
      silence_window_frames_(0),
      silence_threshold_frames_(0),
      left_padding_samples_(0),
      right_padding_samples_(0),
      max_speech_samples_(0),
      window_(1) {
    if (samples_per_ms_ <= 0)
        throw std::runtime_error("sample_rate must be a positive multiple of 1000");
    const int shift_ms = frame_shift_ / samples_per_ms_;
    speech_window_frames_ = ceil_div(config.speech_window_size_ms, shift_ms);
    speech_threshold_frames_ = ceil_div(config.speech_window_threshold_ms, shift_ms);
    silence_window_frames_ = ceil_div(config.silence_window_size_ms, shift_ms);
    silence_threshold_frames_ = ceil_div(config.silence_window_threshold_ms, shift_ms);
    left_padding_samples_ = config.left_padding_ms * samples_per_ms_;
    right_padding_samples_ = config.right_padding_ms * samples_per_ms_;
    max_speech_samples_ = config.max_speech_ms * samples_per_ms_;
    window_ = SlidingWindowBit(std::max(speech_window_frames_, silence_window_frames_));
}

void Segmenter::reset() {
    window_.reset();
    start_ = -1;
    current_ = 0;
    received_samples_ = 0;
    last_end_ = 0;
    seg_idx_ = 0;
    segs_.clear();
}

std::vector<VadSegment> Segmenter::take() {
    std::vector<VadSegment> out = std::move(segs_);
    segs_.clear();
    return out;
}

void Segmenter::push(const float *probs, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        update_frame_state(probs[i]);
        current_ += frame_shift_;
        // Force a split when a segment gets too long.
        if (start_ != -1 && current_ - start_ > max_speech_samples_) {
            on_voice_end(current_);
            on_voice_start();
        }
    }
}

void Segmenter::update_frame_state(float prob) {
    window_.push(prob > threshold_);
    if (start_ == -1) {
        if (window_.check_speech(speech_window_frames_) >= size_t(speech_threshold_frames_))
            on_voice_start();
    } else if (window_.check_silence(silence_window_frames_) >= size_t(silence_threshold_frames_)) {
        on_voice_end(current_);
    }
}

void Segmenter::on_voice_start() {
    // Start = now - run of speech frames - padding, but never before the previous segment's end.
    const int lookback = int(window_.num_right_ones()) * frame_shift_;
    start_ = std::max(last_end_, current_ - lookback - left_padding_samples_);
    VadSegment seg;
    seg.idx = seg_idx_;
    seg.start = start_;
    seg.start_ms = start_ / samples_per_ms_;
    segs_.push_back(seg);
}

void Segmenter::on_voice_end(int end_limit_samples) {
    const int lookback = int(window_.num_right_zeros()) * frame_shift_;
    const int end = std::min(current_ - lookback + right_padding_samples_, end_limit_samples);

    if (!segs_.empty() && segs_.back().end == -1) {
        // The segment was started in this same take() interval: complete it in place.
        segs_.back().end = end;
        segs_.back().end_ms = end / samples_per_ms_;
    } else {
        segs_.emplace_back(seg_idx_, start_, end, start_ / samples_per_ms_, end / samples_per_ms_);
    }
    last_end_ = end;
    start_ = -1;
    seg_idx_++;
}

VadSegment Segmenter::flush() {
    if (start_ != -1) {
        on_voice_end(std::max(current_, received_samples_));
        return segs_.back();
    }
    return VadSegment();
}

} // namespace VadFilterOnnx
