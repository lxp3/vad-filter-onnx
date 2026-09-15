#include "vad/pulse-vad-model.h"

#include <array>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace VadFilterOnnx {

namespace {
constexpr int kSampleRate = 16000;
constexpr int kWindowSamples = 3200;
constexpr int kHopSamples = 1600;
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

std::unique_ptr<VadModel> PulseVadModel::init(const VadConfig &config) {
    if (config.sample_rate != kSampleRate) {
        throw std::runtime_error("PulseVAD supports 16000 Hz audio only");
    }
    auto instance = std::make_unique<PulseVadModel>(*this, config, kHopSamples, kWindowSamples);
    instance->reset();
    return instance;
}

void PulseVadModel::init_state() {}

float PulseVadModel::forward(float *data, int n) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    std::array<int64_t, 2> speech_shape = { 1, n };
    Ort::Value speech =
        Ort::Value::CreateTensor(memory_info, data, n, speech_shape.data(), speech_shape.size());

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(speech));

    auto out = session_->Run(Ort::RunOptions{ nullptr }, input_names_.data(), inputs.data(),
                             inputs.size(), output_names_.data(), output_names_.size());
    return out[0].GetTensorData<float>()[0];
}

} // namespace VadFilterOnnx
