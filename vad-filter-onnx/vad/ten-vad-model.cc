#include "vad/ten-vad-model.h"
#include "utils/onnx-common.h"
#include <string_view>

namespace VadFilterOnnx {

bool is_ten_vad(const std::vector<const char *> &input_names,
                const std::vector<const char *> &output_names) {
    // 7 inputs:  samples, pitch, h1, c1, h2, c2, cache_features
    // 6 outputs: prob, h1, c1, h2, c2, cache_features
    //
    // Match on names rather than counts alone: a bare arity check would also
    // accept unrelated models with the same shape of I/O.
    if (input_names.size() != 7 || output_names.size() != 6) {
        return false;
    }
    return std::string_view(input_names[0]) == "samples" &&
           std::string_view(input_names[1]) == "pitch";
}

void TenVadModel::init_state() {
    if (h1_ == nullptr) {
        h1_ = Ort::Value::CreateTensor<float>(allocator_, state_shape_.data(), state_shape_.size());
        c1_ = Ort::Value::CreateTensor<float>(allocator_, state_shape_.data(), state_shape_.size());
        h2_ = Ort::Value::CreateTensor<float>(allocator_, state_shape_.data(), state_shape_.size());
        c2_ = Ort::Value::CreateTensor<float>(allocator_, state_shape_.data(), state_shape_.size());
        pitch_ =
            Ort::Value::CreateTensor<float>(allocator_, pitch_shape_.data(), pitch_shape_.size());
        conv_cache_ =
            Ort::Value::CreateTensor<float>(allocator_, cache_shape_.data(), cache_shape_.size());
    }
    Fill<float>(&h1_, 0.0f);
    Fill<float>(&c1_, 0.0f);
    Fill<float>(&h2_, 0.0f);
    Fill<float>(&c2_, 0.0f);
    Fill<float>(&pitch_, 0.0f);
    Fill<float>(&conv_cache_, 0.0f);

    if (pitch_estimator_ == nullptr) {
        pitch_estimator_ = std::make_unique<PitchEstimator>(config_.sample_rate, frame_shift_);
    }
    pitch_estimator_->reset();
}

std::unique_ptr<VadModel> TenVadModel::init(const VadConfig &config) {
    // Stride 256, Window 768
    auto instance = std::make_unique<TenVadModel>(*this, config, 256, 768);
    instance->reset();
    return instance;
}

float TenVadModel::forward(float *data, int n) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    std::array<int64_t, 2> x_shape = { 1, n };
    Ort::Value x = Ort::Value::CreateTensor(memory_info, data, n, x_shape.data(), x_shape.size());

    // F0 in Hz for this frame, 0.0 when unvoiced.
    pitch_.GetTensorMutableData<float>()[0] = pitch_estimator_->process(data, n);

    std::vector<Ort::Value> inputs;
    inputs.reserve(7);

    // Inputs: samples, pitch, h1, c1, h2, c2, cache_features
    inputs.push_back(std::move(x));
    inputs.push_back(std::move(pitch_));
    inputs.push_back(std::move(h1_));
    inputs.push_back(std::move(c1_));
    inputs.push_back(std::move(h2_));
    inputs.push_back(std::move(c2_));
    inputs.push_back(std::move(conv_cache_));

    auto out = session_->Run(Ort::RunOptions{ nullptr }, input_names_.data(), inputs.data(),
                             inputs.size(), output_names_.data(), output_names_.size());

    // Reclaim the pitch tensor so it can be reused next frame.
    pitch_ = std::move(inputs[1]);

    // Outputs: prob, h1, c1, h2, c2, cache_features
    h1_ = std::move(out[1]);
    c1_ = std::move(out[2]);
    h2_ = std::move(out[3]);
    c2_ = std::move(out[4]);
    conv_cache_ = std::move(out[5]);

    float prob = out[0].GetTensorData<float>()[0];
    return prob;
}

} // namespace VadFilterOnnx
