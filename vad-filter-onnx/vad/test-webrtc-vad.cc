#include "vad-filter-onnx-cxx-api.h"
#include "vad/webrtc-vad-model.h"

#include <cmath>
#include <functional>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

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

struct FvadDeleter {
    void operator()(Fvad *inst) const {
        if (inst != nullptr) {
            fvad_free(inst);
        }
    }
};

void ExpectThrow(const char *name, const std::function<void()> &fn) {
    try {
        fn();
    } catch (const std::exception &error) {
        std::cout << name << ": caught " << error.what() << '\n';
        return;
    }
    throw std::runtime_error(std::string(name) + " did not throw");
}

} // namespace

int main() {
    using namespace VadFilterOnnx;

    VadConfig config;
    config.sample_rate = 16000;
    config.webrtc_vad_mode = 3;
    config.webrtc_frame_ms = 30;

    auto handle = VadModel::create_webrtc();
    if (handle == nullptr) {
        throw std::runtime_error("create_webrtc returned null");
    }
    auto instance = handle->init(config);
    auto *model = static_cast<WebrtcVadModel *>(instance.get());

    std::unique_ptr<Fvad, FvadDeleter> ref(fvad_new());
    if (ref == nullptr) {
        throw std::runtime_error("fvad_new failed");
    }
    fvad_reset(ref.get());
    if (fvad_set_sample_rate(ref.get(), config.sample_rate) != 0 ||
        fvad_set_mode(ref.get(), config.webrtc_vad_mode) != 0) {
        throw std::runtime_error("failed to configure reference fvad");
    }

    const int frame_samples = WebrtcVadFrameSamples(config);
    std::mt19937 generator(20260915);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    const int num_frames = 200;
    float max_diff = 0.0f;
    int mismatches = 0;

    std::vector<float> frame(static_cast<size_t>(frame_samples));
    std::vector<int16_t> pcm(static_cast<size_t>(frame_samples));
    for (int i = 0; i < num_frames; ++i) {
        for (int n = 0; n < frame_samples; ++n) {
            frame[static_cast<size_t>(n)] = distribution(generator);
            pcm[static_cast<size_t>(n)] = FloatToPcm16(frame[static_cast<size_t>(n)]);
        }
        const float got = model->forward(frame.data(), frame_samples);
        const int expected = fvad_process(ref.get(), pcm.data(), static_cast<size_t>(frame_samples));
        if (expected < 0) {
            throw std::runtime_error("reference fvad_process failed");
        }
        const float want = expected > 0 ? 1.0f : 0.0f;
        const float diff = std::fabs(got - want);
        if (diff > max_diff) {
            max_diff = diff;
        }
        if (diff > 0.0f) {
            ++mismatches;
        }
    }

    std::cout << std::fixed;
    std::cout << "frames " << num_frames << '\n';
    std::cout << "max_diff " << max_diff << '\n';
    std::cout << "mismatches " << mismatches << '\n';
    if (max_diff != 0.0f || mismatches != 0) {
        throw std::runtime_error("WebRTC VAD frame decisions differ from libfvad");
    }

    auto api_handle = AutoVadModel::create_webrtc();
    auto api_model = api_handle->init(config);
    VadConfig updated = api_model->get_config();
    if (updated.webrtc_vad_mode != 3 || updated.webrtc_frame_ms != 30) {
        throw std::runtime_error("unexpected default webrtc config");
    }
    updated.webrtc_vad_mode = 2;
    updated.webrtc_frame_ms = 10;
    api_model->setup_config(updated);
    updated = api_model->get_config();
    if (updated.webrtc_vad_mode != 2 || updated.webrtc_frame_ms != 10) {
        throw std::runtime_error("setup_config did not apply webrtc fields");
    }

    VadConfig bad = api_model->get_config();
    bad.sample_rate = 8000;
    ExpectThrow("sample_rate change", [&] { api_model->setup_config(bad); });

    VadConfig invalid_rate;
    invalid_rate.sample_rate = 22050;
    ExpectThrow("invalid sample_rate", [&] { handle->init(invalid_rate); });

    VadConfig invalid_mode = config;
    invalid_mode.webrtc_vad_mode = 4;
    ExpectThrow("invalid mode", [&] { handle->init(invalid_mode); });

    VadConfig invalid_frame = config;
    invalid_frame.webrtc_frame_ms = 25;
    ExpectThrow("invalid frame_ms", [&] { handle->init(invalid_frame); });

    std::cout << "WebRTC VAD checks passed\n";
    return 0;
}
