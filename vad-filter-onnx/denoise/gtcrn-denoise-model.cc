#include "denoise/gtcrn-denoise-model.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <string_view>

namespace VadFilterOnnx {
namespace {

constexpr std::size_t kHopSize = 256;
constexpr std::size_t kConvCacheSize = 2 * 1 * 16 * 16 * 33;
constexpr std::size_t kTraCacheSize = 2 * 3 * 1 * 1 * 16;
constexpr std::size_t kInterCacheSize = 2 * 1 * 33 * 16;

constexpr std::array<const char *, 6> kInputNames = {
    "speech", "conv_cache", "tra_cache", "inter_cache", "analysis_cache", "synthesis_cache",
};
constexpr std::array<const char *, 6> kOutputNames = {
    "enhanced",        "conv_cache_out",     "tra_cache_out",
    "inter_cache_out", "analysis_cache_out", "synthesis_cache_out",
};

} // namespace

bool is_gtcrn_denoise(const std::vector<const char *> &input_names,
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

GtcrnDenoiseModel::GtcrnDenoiseModel(const GtcrnDenoiseModel &other, const DenoiseConfig &config)
    : config_(config) {
    session_ = other.session_;
    input_names_ = other.input_names_;
    output_names_ = other.output_names_;
    reset();
}

std::unique_ptr<DenoiseModel> GtcrnDenoiseModel::init(const DenoiseConfig &config) {
    if (config.sample_rate != 16000) {
        throw std::invalid_argument("GTCRN only supports a 16000 Hz sample rate");
    }
    return std::unique_ptr<DenoiseModel>(new GtcrnDenoiseModel(*this, config));
}

void GtcrnDenoiseModel::reset() {
    conv_cache_.assign(kConvCacheSize, 0.0F);
    tra_cache_.assign(kTraCacheSize, 0.0F);
    inter_cache_.assign(kInterCacheSize, 0.0F);
    analysis_cache_.assign(kHopSize, 0.0F);
    synthesis_cache_.assign(kHopSize, 0.0F);
    input_buffer_.clear();
    input_offset_ = 0;
    received_samples_ = 0;
    emitted_samples_ = 0;
    primed_ = false;
    finished_ = false;
}

std::vector<float> GtcrnDenoiseModel::forward(const float *speech) {
    const std::array<int64_t, 2> speech_shape = { 1, 256 };
    const std::array<int64_t, 5> conv_shape = { 2, 1, 16, 16, 33 };
    const std::array<int64_t, 5> tra_shape = { 2, 3, 1, 1, 16 };
    const std::array<int64_t, 4> inter_shape = { 2, 1, 33, 16 };
    const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> inputs;
    inputs.reserve(6);
    inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, const_cast<float *>(speech),
                                                     kHopSize, speech_shape.data(),
                                                     speech_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(
        memory_info, conv_cache_.data(), conv_cache_.size(), conv_shape.data(), conv_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(
        memory_info, tra_cache_.data(), tra_cache_.size(), tra_shape.data(), tra_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, inter_cache_.data(),
                                                     inter_cache_.size(), inter_shape.data(),
                                                     inter_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, analysis_cache_.data(),
                                                     analysis_cache_.size(), speech_shape.data(),
                                                     speech_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, synthesis_cache_.data(),
                                                     synthesis_cache_.size(), speech_shape.data(),
                                                     speech_shape.size()));

    auto outputs = session_->Run(Ort::RunOptions{ nullptr }, input_names_.data(), inputs.data(),
                                 inputs.size(), output_names_.data(), output_names_.size());
    std::vector<float> enhanced(kHopSize);
    std::copy_n(outputs[0].GetTensorData<float>(), kHopSize, enhanced.data());
    std::copy_n(outputs[1].GetTensorData<float>(), conv_cache_.size(), conv_cache_.data());
    std::copy_n(outputs[2].GetTensorData<float>(), tra_cache_.size(), tra_cache_.data());
    std::copy_n(outputs[3].GetTensorData<float>(), inter_cache_.size(), inter_cache_.data());
    std::copy_n(outputs[4].GetTensorData<float>(), analysis_cache_.size(), analysis_cache_.data());
    std::copy_n(outputs[5].GetTensorData<float>(), synthesis_cache_.size(),
                synthesis_cache_.data());
    return enhanced;
}

void GtcrnDenoiseModel::compact_input() {
    if (input_offset_ == 0) {
        return;
    }
    if (input_offset_ == input_buffer_.size()) {
        input_buffer_.clear();
    } else {
        std::vector<float> remaining(input_buffer_.begin() + input_offset_, input_buffer_.end());
        input_buffer_.swap(remaining);
    }
    input_offset_ = 0;
}

std::vector<float> GtcrnDenoiseModel::decode(const float *data, int n, bool input_finished) {
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
    received_samples_ += static_cast<std::size_t>(n);
    std::vector<float> result;

    auto process_hop = [&](const float *hop) {
        auto enhanced = forward(hop);
        if (primed_) {
            result.insert(result.end(), enhanced.begin(), enhanced.end());
        } else {
            primed_ = true;
        }
    };

    while (input_buffer_.size() - input_offset_ >= kHopSize) {
        process_hop(input_buffer_.data() + input_offset_);
        input_offset_ += kHopSize;
    }
    compact_input();

    if (input_finished) {
        if (!input_buffer_.empty()) {
            std::array<float, kHopSize> padded{};
            std::copy(input_buffer_.begin(), input_buffer_.end(), padded.begin());
            process_hop(padded.data());
        }
        if (primed_) {
            const std::array<float, kHopSize> zeros{};
            process_hop(zeros.data());
        }
        input_buffer_.clear();
        input_offset_ = 0;
        finished_ = true;
    }

    const std::size_t available = received_samples_ - emitted_samples_;
    if (result.size() > available) {
        result.resize(available);
    }
    emitted_samples_ += result.size();
    return result;
}

} // namespace VadFilterOnnx
