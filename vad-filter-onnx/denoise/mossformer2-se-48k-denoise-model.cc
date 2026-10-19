#include "denoise/mossformer2-se-48k-denoise-model.h"
#include <algorithm>
#include <array>
#include <stdexcept>
#include <string_view>

namespace VadFilterOnnx {
namespace {

constexpr std::array<const char *, 1> kInputNames = { "speech" };
constexpr std::array<const char *, 1> kOutputNames = { "enhanced" };

// Matches ClearerVoice-Studio's MossFormer2_SE_48K inference config:
// one_time_decode_length=20 (seconds), decode_window=4 (seconds).
constexpr double kOneTimeDecodeLengthSeconds = 20.0;
constexpr double kDecodeWindowSeconds = 4.0;
constexpr double kSegmentStrideRatio = 0.75;

} // namespace

bool is_mossformer2_se_48k_denoise(const std::vector<const char *> &input_names,
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

Mossformer2Se48kDenoiseModel::Mossformer2Se48kDenoiseModel(
    const Mossformer2Se48kDenoiseModel &other, const DenoiseConfig &config)
    : config_(config), sample_rate_(other.sample_rate_) {
    session_ = other.session_;
    input_names_ = other.input_names_;
    output_names_ = other.output_names_;
    reset();
}

std::unique_ptr<DenoiseModel> Mossformer2Se48kDenoiseModel::init(const DenoiseConfig &config) {
    if (config.sample_rate != sample_rate_) {
        throw std::invalid_argument("MossFormer2_SE_48K model only supports a " +
                                    std::to_string(sample_rate_) + " Hz sample rate");
    }
    return std::unique_ptr<DenoiseModel>(new Mossformer2Se48kDenoiseModel(*this, config));
}

void Mossformer2Se48kDenoiseModel::reset() {
    input_buffer_.clear();
    finished_ = false;
}

std::vector<float> Mossformer2Se48kDenoiseModel::forward(const float *data, std::size_t n) {
    const std::array<int64_t, 2> speech_shape = { 1, static_cast<int64_t>(n) };
    const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float *>(data), n, speech_shape.data(), speech_shape.size()));

    auto outputs = session_->Run(Ort::RunOptions{ nullptr }, input_names_.data(), inputs.data(),
                                 inputs.size(), output_names_.data(), output_names_.size());
    const auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const std::size_t out_len = out_shape.empty() ? 0 : static_cast<std::size_t>(out_shape.back());
    std::vector<float> enhanced(out_len);
    std::copy_n(outputs[0].GetTensorData<float>(), out_len, enhanced.data());
    return enhanced;
}

std::vector<float> Mossformer2Se48kDenoiseModel::decode_segmented() {
    const std::size_t t = input_buffer_.size();
    const std::size_t one_time_decode_length =
        static_cast<std::size_t>(sample_rate_ * kOneTimeDecodeLengthSeconds);
    if (t <= one_time_decode_length) {
        return forward(input_buffer_.data(), t);
    }

    const std::size_t window = static_cast<std::size_t>(sample_rate_ * kDecodeWindowSeconds);
    const std::size_t stride = static_cast<std::size_t>(window * kSegmentStrideRatio);
    const std::size_t give_up = (window - stride) / 2;

    std::vector<float> outputs(t, 0.0F);
    std::size_t current_idx = 0;
    while (current_idx + window <= t) {
        auto segment = forward(input_buffer_.data() + current_idx, window);
        if (current_idx == 0) {
            std::copy_n(segment.begin(), window - give_up, outputs.begin());
        } else {
            std::copy(segment.begin() + give_up, segment.end() - give_up,
                      outputs.begin() + current_idx + give_up);
        }
        current_idx += stride;
    }
    if (current_idx < t) {
        const std::size_t last_start = current_idx - give_up;
        auto segment = forward(input_buffer_.data() + last_start, t - last_start);
        std::copy(segment.begin() + give_up, segment.end(), outputs.begin() + current_idx);
    }
    return outputs;
}

std::vector<float> Mossformer2Se48kDenoiseModel::decode(const float *data, int n,
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
    return decode_segmented();
}

} // namespace VadFilterOnnx
