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
        std::string_view(input_names[1]) == "state" &&
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
    // Silero V4 uses fixed 512 samples
    auto instance = std::make_unique<SileroVadModelV4>(*this, config, 512, 512);
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
    // Silero V5: shift is 256/512, length adds context (32/64)
    int frame_shift = (config.sample_rate == 8000 ? 256 : 512);
    int context_size = (config.sample_rate == 8000 ? 32 : 64);
    int frame_length = frame_shift + context_size;
    auto instance = std::make_unique<SileroVadModelV5>(*this, config, frame_shift, frame_length);
    instance->reset();
    return instance;
}

float SileroVadModelV5::forward(float *data, int n) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    std::array<int64_t, 2> x_shape = { 1, n };
    Ort::Value x = Ort::Value::CreateTensor(memory_info, data, n, x_shape.data(), x_shape.size());

    // 使用 vector 替代 array，解决 Ort::Value 无法默认构造的问题
    std::vector<Ort::Value> inputs;
    inputs.reserve(input_names_.size());

    // 基础输入：input 和 state
    inputs.push_back(std::move(x));
    inputs.push_back(std::move(state_));

    // 如果模型需要采样率输入 (size 为 3)
    if (input_names_.size() > 2) {
        int64_t sr_shape = 1;
        int64_t sample_rate = static_cast<int64_t>(config_.sample_rate);
        Ort::Value sr = Ort::Value::CreateTensor(memory_info, &sample_rate, 1, &sr_shape, 1);
        inputs.push_back(std::move(sr));
    }

    auto out = session_->Run(Ort::RunOptions{ nullptr }, input_names_.data(), inputs.data(),
                             inputs.size(), output_names_.data(), output_names_.size());

    state_ = std::move(out[1]);
    float prob = out[0].GetTensorData<float>()[0];
    return prob;
}

} // namespace VadFilterOnnx
