#pragma once

#include "sliding-window-bit.h"
#include "vad-config.h"
#include <vector>

namespace VadFilterOnnx {

// Turns a stream of per-frame speech probabilities into speech segments. Independent of the
// acoustic model: it only needs the frame shift, the decision threshold and the VadConfig.
class Segmenter {
  public:
    Segmenter(const VadConfig &config, int frame_shift_samples, float threshold);

    const VadConfig &config() const { return config_; }

    // Records that n more input samples were received (used to close a segment at flush()).
    void add_samples(int n) { received_samples_ += n; }

    // Consumes the probabilities of n consecutive frames.
    void push(const float *probs, size_t n);

    // Closes the open segment, if any, at the end of the stream. Returns it (or an empty segment).
    VadSegment flush();

    // Returns the segments completed or started since the last call.
    std::vector<VadSegment> take();

    void reset();

  private:
    void update_frame_state(float prob);
    void on_voice_start();
    void on_voice_end(int end_limit_samples);

    VadConfig config_;
    int frame_shift_;
    float threshold_;
    int samples_per_ms_;
    int speech_window_frames_;
    int speech_threshold_frames_;
    int silence_window_frames_;
    int silence_threshold_frames_;
    int left_padding_samples_;
    int right_padding_samples_;
    int max_speech_samples_;
    SlidingWindowBit window_;

    int start_ = -1; // speech start position in samples, -1 while in silence
    int current_ = 0;
    int received_samples_ = 0;
    int last_end_ = 0;
    int seg_idx_ = 0;
    std::vector<VadSegment> segs_;
};

} // namespace VadFilterOnnx
