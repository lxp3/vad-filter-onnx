#include "denoise/frcrn-se-16k-denoise-model.h"
#include <algorithm>
#include <array>
#include <stdexcept>
#include <string_view>

namespace VadFilterOnnx {
namespace {

constexpr std::array<const char *, 1> kInputNames = { "speech" };
constexpr std::array<const char *, 1> kOutputNames = { "enhanced" };

} // namespace

bool is_frcrn_se_16k_denoise(const std::vector<const char *> &input_names,
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

FrcrnSe16kDenoiseModel::FrcrnSe16kDenoiseModel(const FrcrnSe16kDenoiseModel &other, const DenoiseConfig &config)
    : config_(config), sample_rate_(other.sample_rate_) {
    session_ = other.session_;
    input_names_ = other.input_names_;
    output_names_ = other.output_names_;
    reset();
}

std::unique_ptr<DenoiseModel> FrcrnSe16kDenoiseModel::init(const DenoiseConfig &config) {
    if (config.sample_rate != sample_rate_) {
        throw std::invalid_argument("FRCRN model only supports a " +
                                    std::to_string(sample_rate_) + " Hz sample rate");
    }
    return std::unique_ptr<DenoiseModel>(new FrcrnSe16kDenoiseModel(*this, config));
}

void FrcrnSe16kDenoiseModel::reset() {
    input_buffer_.clear();
    finished_ = false;
}

std::vector<float> FrcrnSe16kDenoiseModel::forward() {
    const std::size_t original = input_buffer_.size();
    // STFT in the exported graph is win=40ms, hop=20ms. Lengths shorter
    // than one window, or not hop-aligned, either fail Conv or return a
    // truncated waveform. Pad zeros for the session, then crop back.
    const std::size_t win = static_cast<std::size_t>(std::max(sample_rate_ * 40 / 1000, 1));
    const std::size_t hop = static_cast<std::size_t>(std::max(sample_rate_ * 20 / 1000, 1));
    std::size_t padded = original;
    if (padded < win) {
        padded = win;
    } else if (padded % hop != 0) {
        padded = (padded / hop + 1) * hop;
    }
    std::vector<float> speech = input_buffer_;
    if (speech.size() < padded) {
        speech.resize(padded, 0.0f);
    }

    const std::array<int64_t, 2> speech_shape = { 1, static_cast<int64_t>(speech.size()) };
    const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, speech.data(), speech.size(),
                                                      speech_shape.data(), speech_shape.size()));

    auto outputs = session_->Run(Ort::RunOptions{ nullptr }, input_names_.data(), inputs.data(),
                                 inputs.size(), output_names_.data(), output_names_.size());
    const auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const std::size_t out_len = out_shape.empty() ? 0 : static_cast<std::size_t>(out_shape.back());
    std::vector<float> enhanced(original, 0.0f);
    const std::size_t copy = std::min(original, out_len);
    if (copy > 0) {
        std::copy_n(outputs[0].GetTensorData<float>(), copy, enhanced.data());
    }
    return enhanced;
}

std::vector<float> FrcrnSe16kDenoiseModel::decode(const float *data, int n, bool input_finished) {
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
        if (sample_rate_ <= 0) {
            throw std::runtime_error("FRCRN sample rate is not initialized");
        }
        const auto max_samples = static_cast<std::size_t>(sample_rate_);
        if (static_cast<std::size_t>(n) > max_samples ||
            input_buffer_.size() + static_cast<std::size_t>(n) > max_samples) {
            throw std::invalid_argument("FRCRN decode input exceeds 1 second (max " +
                                        std::to_string(max_samples) + " samples)");
        }
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
