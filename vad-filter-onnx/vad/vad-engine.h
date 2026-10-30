#pragma once

#include "vad-config.h"
#include <vector>

namespace VadFilterOnnx {

// The model side of a VAD: turns samples into one speech probability per frame. Knows nothing
// about segments; VadStream feeds its output to a Segmenter.
class VadEngine {
  public:
    virtual ~VadEngine() = default;

    // Number of samples each probability advances the stream by.
    virtual int frame_shift() const = 0;

    // Probability above which a frame counts as speech.
    virtual float threshold(const VadConfig &config) const { return config.threshold; }

    // New stream: clear recurrent state and any buffered samples.
    virtual void reset() = 0;

    // Appends n samples and the probabilities of the frames that became complete to `probs`.
    // With finished = true all remaining samples are consumed.
    virtual void process(float *data, int n, bool finished, std::vector<float> &probs) = 0;
};

// Engine for models that score one fixed-size window per call (Silero, Ten, Pulse, WebRTC).
// Windows of frame_length samples start every frame_shift samples.
class FrameEngine : public VadEngine {
  public:
    FrameEngine(int frame_shift, int frame_length)
        : frame_shift_(frame_shift), frame_length_(frame_length) {}

    int frame_shift() const override { return frame_shift_; }

    void reset() final {
        reminder_.clear();
        reset_state();
    }

    void process(float *data, int n, bool finished, std::vector<float> &probs) final {
        float *ptr = data;
        int len = n;
        if (!reminder_.empty()) {
            reminder_.insert(reminder_.end(), data, data + n);
            ptr = reminder_.data();
            len = static_cast<int>(reminder_.size());
        }

        while (len >= frame_length_) {
            probs.push_back(forward(ptr));
            ptr += frame_shift_;
            len -= frame_shift_;
        }

        // Keep the unconsumed tail (including the overlap needed by the next window).
        if (finished || len <= 0) {
            reminder_.clear();
        } else if (reminder_.empty()) {
            reminder_.assign(ptr, ptr + len);
        } else {
            std::vector<float> next(ptr, ptr + len);
            reminder_ = std::move(next);
        }
    }

  protected:
    // Scores one window of frame_length samples.
    virtual float forward(float *frame) = 0;
    // Zeroes the model state.
    virtual void reset_state() = 0;

    int frame_shift_;
    int frame_length_;

  private:
    std::vector<float> reminder_;
};

} // namespace VadFilterOnnx
