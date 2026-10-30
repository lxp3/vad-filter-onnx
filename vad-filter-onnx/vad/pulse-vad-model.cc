#include "vad/pulse-vad-model.h"

#include <stdexcept>
#include <string_view>

namespace VadFilterOnnx {

namespace {

constexpr int kSampleRate = 16000;
constexpr int kWindowSamples = 3200;
constexpr int kHopSamples = 1600;

class PulseEngine final : public FrameEngine {
  public:
    explicit PulseEngine(std::shared_ptr<const OnnxModel> model)
        : FrameEngine(kHopSamples, kWindowSamples), ort_(std::move(model)) {}

  private:
    void reset_state() override {}

    float forward(float *data) override {
        std::vector<Ort::Value> inputs;
        inputs.push_back(ort_.wave(data, frame_length_));
        auto out = ort_.run(inputs);
        return out[0].GetTensorData<float>()[0];
    }

    OrtRunner ort_;
};

} // namespace

bool is_pulse_vad(Ort::Session *session) {
    if (session == nullptr) {
        return false;
    }
    Ort::AllocatorWithDefaultOptions allocator;
    auto model_type =
        session->GetModelMetadata().LookupCustomMetadataMapAllocated("model_type", allocator);
    return model_type && std::string_view(model_type.get()) == "pulsevad";
}

std::unique_ptr<VadEngine> make_pulse_engine(std::shared_ptr<const OnnxModel> model,
                                             const VadConfig &config) {
    if (config.sample_rate != kSampleRate) {
        throw std::runtime_error("PulseVAD supports 16000 Hz audio only");
    }
    return std::make_unique<PulseEngine>(std::move(model));
}

} // namespace VadFilterOnnx
