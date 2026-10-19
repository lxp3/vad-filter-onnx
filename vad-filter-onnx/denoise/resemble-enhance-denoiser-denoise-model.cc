#include "denoise/resemble-enhance-denoiser-denoise-model.h"
#include <array>
#include <stdexcept>
#include <string_view>

namespace VadFilterOnnx {
namespace {

constexpr std::array<const char *, 1> kInputNames = { "speech" };
constexpr std::array<const char *, 1> kOutputNames = { "enhanced" };

} // namespace

bool is_resemble_enhance_denoiser_denoise(const std::vector<const char *> &input_names,
                                          const std::vector<const char *> &output_names) {
    if (input_names.size() != kInputNames.size() || output_names.size() != kOutputNames.size()) {
        return false;
    }
    for (std::size_t index = 0; index < kInputNames.size(); ++index) {
        if (std::string_view(input_names[index]) != kInputNames[index] ||
            std::string_view(output_names[index]) != kOutputNames[index]) {
            return false;
        }
    }
    return true;
}

ResembleEnhanceDenoiserDenoiseModel::ResembleEnhanceDenoiserDenoiseModel(
    const ResembleEnhanceDenoiserDenoiseModel &other, const DenoiseConfig &config)
    : config_(config), sample_rate_(other.sample_rate_) {
    session_ = other.session_;
    input_names_ = other.input_names_;
    output_names_ = other.output_names_;
    reset();
}

std::unique_ptr<DenoiseModel> ResembleEnhanceDenoiserDenoiseModel::init(const DenoiseConfig &config) {
    if (config.sample_rate != sample_rate_) {
        throw std::invalid_argument("resemble-enhance Denoiser model only supports a " +
                                    std::to_string(sample_rate_) + " Hz sample rate");
    }
    return std::unique_ptr<DenoiseModel>(new ResembleEnhanceDenoiserDenoiseModel(*this, config));
}

void ResembleEnhanceDenoiserDenoiseModel::reset() {
    input_buffer_.clear();
    finished_ = false;
}

std::vector<float> ResembleEnhanceDenoiserDenoiseModel::forward() {
    const std::array<int64_t, 2> speech_shape = { 1, static_cast<int64_t>(input_buffer_.size()) };
    const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, input_buffer_.data(),
                                                      input_buffer_.size(), speech_shape.data(),
                                                      speech_shape.size()));

    auto outputs = session_->Run(Ort::RunOptions{ nullptr }, input_names_.data(), inputs.data(),
                                 inputs.size(), output_names_.data(), output_names_.size());
    const auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const std::size_t out_len = out_shape.empty() ? 0 : static_cast<std::size_t>(out_shape.back());
    std::vector<float> enhanced(out_len);
    std::copy_n(outputs[0].GetTensorData<float>(), out_len, enhanced.data());
    return enhanced;
}

std::vector<float> ResembleEnhanceDenoiserDenoiseModel::decode(const float *data, int n,
                                                                bool input_finished) {
    if (n < 0 || (n > 0 && data == nullptr)) {
        throw std::invalid_argument("Invalid denoise input buffer");
    }
    if (finished_) {
        if (n == 0 && input_finished) {
            return {};
        }
        throw std::runtime_error("Denoise stream is finished; call reset() first");
    }

    if (n > 0) {
        input_buffer_.insert(input_buffer_.end(), data, data + n);
    }

    if (!input_finished) {
        // Non-streaming model: input may be fed incrementally, but no
        // output is ever produced before the stream is marked finished.
        return {};
    }

    finished_ = true;
    if (input_buffer_.empty()) {
        return {};
    }
    return forward();
}

} // namespace VadFilterOnnx
