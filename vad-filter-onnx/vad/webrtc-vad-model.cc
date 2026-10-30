#include "vad/webrtc-vad-model.h"

extern "C" {
#include <fvad.h>
}
#include <cstdint>
#include <vector>

#include <cmath>
#include <stdexcept>
#include <string>

namespace VadFilterOnnx {

namespace {

int16_t FloatToPcm16(float sample) {
    float scaled = sample * 32768.0f;
    if (scaled > 32767.0f) {
        scaled = 32767.0f;
    } else if (scaled < -32768.0f) {
        scaled = -32768.0f;
    }
    return static_cast<int16_t>(std::lrintf(scaled));
}

bool IsValidWebrtcSampleRate(int sample_rate) {
    return sample_rate == 8000 || sample_rate == 16000 || sample_rate == 32000 ||
           sample_rate == 48000;
}

struct FvadDeleter {
    void operator()(Fvad *inst) const {
        if (inst != nullptr) {
            fvad_free(inst);
        }
    }
};

class WebrtcEngine final : public FrameEngine {
  public:
    WebrtcEngine(const VadConfig &config, int frame_samples)
        : FrameEngine(frame_samples, frame_samples),
          fvad_(fvad_new()),
          sample_rate_(config.sample_rate),
          mode_(config.webrtc_vad_mode) {
        if (fvad_ == nullptr) {
            throw std::runtime_error("Failed to create WebRTC VAD instance");
        }
        reset_state();
    }

    // fvad_process returns 0 or 1, so 0.5 separates the two whatever VadConfig::threshold is.
    float threshold(const VadConfig &) const override { return 0.5f; }

  private:
    void reset_state() override {
        fvad_reset(fvad_.get());
        if (fvad_set_sample_rate(fvad_.get(), sample_rate_) != 0) {
            throw std::runtime_error("fvad_set_sample_rate failed for " +
                                     std::to_string(sample_rate_));
        }
        if (fvad_set_mode(fvad_.get(), mode_) != 0) {
            throw std::runtime_error("fvad_set_mode failed for " + std::to_string(mode_));
        }
    }

    float forward(float *data) override {
        const int n = frame_length_;
        pcm16_.resize(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            pcm16_[static_cast<size_t>(i)] = FloatToPcm16(data[i]);
        }
        const int decision = fvad_process(fvad_.get(), pcm16_.data(), static_cast<size_t>(n));
        if (decision < 0) {
            throw std::runtime_error("fvad_process rejected frame length " + std::to_string(n));
        }
        return decision > 0 ? 1.0f : 0.0f;
    }

    std::unique_ptr<Fvad, FvadDeleter> fvad_;
    int sample_rate_;
    int mode_;
    std::vector<int16_t> pcm16_;
};

} // namespace

void ValidateWebrtcVadConfig(const VadConfig &config) {
    if (!IsValidWebrtcSampleRate(config.sample_rate)) {
        throw std::runtime_error(
            "WebRTC VAD sample_rate must be 8000, 16000, 32000, or 48000, got " +
            std::to_string(config.sample_rate));
    }
    if (config.webrtc_vad_mode < 0 || config.webrtc_vad_mode > 3) {
        throw std::runtime_error("WebRTC VAD webrtc_vad_mode must be 0-3, got " +
                                 std::to_string(config.webrtc_vad_mode));
    }
    if (config.webrtc_frame_ms != 10 && config.webrtc_frame_ms != 20 &&
        config.webrtc_frame_ms != 30) {
        throw std::runtime_error("WebRTC VAD webrtc_frame_ms must be 10, 20, or 30, got " +
                                 std::to_string(config.webrtc_frame_ms));
    }
}

int WebrtcVadFrameSamples(const VadConfig &config) {
    ValidateWebrtcVadConfig(config);
    return (config.sample_rate / 1000) * config.webrtc_frame_ms;
}

std::unique_ptr<VadEngine> make_webrtc_engine(const VadConfig &config) {
    return std::make_unique<WebrtcEngine>(config, WebrtcVadFrameSamples(config));
}

} // namespace VadFilterOnnx
