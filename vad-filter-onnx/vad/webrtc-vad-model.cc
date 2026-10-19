#include "vad/webrtc-vad-model.h"

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

std::unique_ptr<VadModel> WebrtcVadModel::init(const VadConfig &config) {
    const int frame_samples = WebrtcVadFrameSamples(config);
    auto instance = std::make_unique<WebrtcVadModel>(*this, config, frame_samples, frame_samples);
    instance->engine_.reset(fvad_new());
    if (instance->engine_ == nullptr) {
        throw std::runtime_error("Failed to create WebRTC VAD instance");
    }
    instance->reset();
    return instance;
}

void WebrtcVadModel::init_state() {
    if (engine_ == nullptr) {
        throw std::runtime_error("WebRTC VAD engine is not initialized");
    }
    fvad_reset(engine_.get());
    if (fvad_set_sample_rate(engine_.get(), config_.sample_rate) != 0) {
        throw std::runtime_error("fvad_set_sample_rate failed for " +
                                 std::to_string(config_.sample_rate));
    }
    if (fvad_set_mode(engine_.get(), config_.webrtc_vad_mode) != 0) {
        throw std::runtime_error("fvad_set_mode failed for " +
                                 std::to_string(config_.webrtc_vad_mode));
    }
}

float WebrtcVadModel::forward(float *data, int n) {
    if (engine_ == nullptr) {
        throw std::runtime_error("WebRTC VAD engine is not initialized");
    }
    if (n != frame_length_) {
        throw std::runtime_error("WebRTC VAD expected " + std::to_string(frame_length_) +
                                 " samples, got " + std::to_string(n));
    }

    pcm16_.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        pcm16_[static_cast<size_t>(i)] = FloatToPcm16(data[i]);
    }

    const int decision = fvad_process(engine_.get(), pcm16_.data(), static_cast<size_t>(n));
    if (decision < 0) {
        throw std::runtime_error("fvad_process rejected frame length " + std::to_string(n));
    }
    return decision > 0 ? 1.0f : 0.0f;
}

} // namespace VadFilterOnnx
