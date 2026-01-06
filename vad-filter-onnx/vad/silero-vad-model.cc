#include "vad/silero-vad-model.h"
#include "utils/onnx-common.h"
#include <string_view>

namespace VadFilterOnnx {

bool is_silero_vad_v4(const std::vector<const char *> &input_names,
                      const std::vector<const char *> &output_names) {
    if (input_names.size() == 4 && output_names.size() == 3 &&
        std::string_view(input_names[0]) == "input" && std::string_view(input_names[1]) == "sr" &&
        std::string_view(input_names[2]) == "h" && std::string_view(input_names[3]) == "c" &&
        std::string_view(output_names[0]) == "output" &&
        std::string_view(output_names[1]) == "hn" && std::string_view(output_names[2]) == "cn") {
        return true;
    }
    return false;
}

bool is_silero_vad_v5(const std::vector<const char *> &input_names,
                      const std::vector<const char *> &output_names) {
    if (input_names.size() == 3 && output_names.size() == 2 &&
        std::string_view(input_names[0]) == "input" &&
        std::string_view(input_names[1]) == "state" && std::string_view(input_names[2]) == "sr" &&
        std::string_view(output_names[0]) == "output" &&
        std::string_view(output_names[1]) == "stateN") {
        return true;
    }
    return false;
}

void SileroVadModelV4::init_state() {
    if (h_state_ == nullptr) {
        h_state_ = Ort::Value::CreateTensor<float>(allocator_, shape_.data(), shape_.size());
        c_state_ = Ort::Value::CreateTensor<float>(allocator_, shape_.data(), shape_.size());
    }
    Fill<float>(&h_state_, 0.0f);
    Fill<float>(&c_state_, 0.0f);
}

std::unique_ptr<VadModel> SileroVadModelV4::init(const VadConfig &config) {
    auto instance = std::make_unique<SileroVadModelV4>(static_cast<const VadModel &>(*this));
    instance->config_ = config;
    instance->frame_shift_ = 512;
    instance->frame_length_ = 512;

    int fs_ms = 1000 * instance->frame_shift_ / config.sample_rate;
    int ws = (config.window_size_ms + fs_ms - 1) / fs_ms;
    int wt = (config.min_speech_ms + fs_ms - 1) / fs_ms;
    instance->window_detector_ = std::make_unique<SlidingWindowBit>(ws, wt);

    instance->reset();
    return instance;
}

float SileroVadModelV4::forward(float *data, int n) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    std::array<int64_t, 2> x_shape = { 1, n };
    Ort::Value x = Ort::Value::CreateTensor(memory_info, data, n, x_shape.data(), x_shape.size());

    int64_t sr_shape = 1;
    int64_t sample_rate = static_cast<int64_t>(config_.sample_rate);
    Ort::Value sr = Ort::Value::CreateTensor(memory_info, &sample_rate, 1, &sr_shape, 1);

    std::array<Ort::Value, 4> inputs = { std::move(x), std::move(sr), std::move(h_state_),
                                         std::move(c_state_) };

    auto out = session_->Run(Ort::RunOptions{ nullptr }, input_names_.data(), inputs.data(),
                             inputs.size(), output_names_.data(), output_names_.size());

    h_state_ = std::move(out[1]);
    c_state_ = std::move(out[2]);
    float prob = out[0].GetTensorData<float>()[0];
    return prob;
}

/* SileroVadModelV5 Implementation */

void SileroVadModelV5::init_state() {
    if (state_ == nullptr) {
        state_ = Ort::Value::CreateTensor<float>(allocator_, shape_.data(), shape_.size());
    }
    Fill<float>(&state_, 0.0f);
}

std::unique_ptr<VadModel> SileroVadModelV5::init(const VadConfig &config) {
    auto instance = std::make_unique<SileroVadModelV5>(static_cast<const VadModel &>(*this));
    instance->config_ = config;
    instance->frame_shift_ = (config.sample_rate == 8000 ? 256 : 512);
    int context_size = (config.sample_rate == 8000 ? 32 : 64);
    instance->frame_length_ = instance->frame_shift_ + context_size;

    int fs_ms = 1000 * instance->frame_shift_ / config.sample_rate;
    int ws = (config.window_size_ms + fs_ms - 1) / fs_ms;
    int wt = (config.min_speech_ms + fs_ms - 1) / fs_ms;
    instance->window_detector_ = std::make_unique<SlidingWindowBit>(ws, wt);

    instance->reset();
    return instance;
}

float SileroVadModelV5::forward(float *data, int n) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    std::array<int64_t, 2> x_shape = { 1, n };
    Ort::Value x = Ort::Value::CreateTensor(memory_info, data, n, x_shape.data(), x_shape.size());

    int64_t sr_shape = 1;
    int64_t sample_rate = static_cast<int64_t>(config_.sample_rate);
    Ort::Value sr = Ort::Value::CreateTensor(memory_info, &sample_rate, 1, &sr_shape, 1);

    std::array<Ort::Value, 3> inputs = { std::move(x), std::move(state_), std::move(sr) };

    auto out = session_->Run(Ort::RunOptions{ nullptr }, input_names_.data(), inputs.data(),
                             inputs.size(), output_names_.data(), output_names_.size());

    state_ = std::move(out[1]);
    float prob = out[0].GetTensorData<float>()[0];
    return prob;
}
} // namespace VadFilterOnnx
