#include "denoise/dfsmn-ans-psm-48k-denoise-model.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <string_view>

namespace VadFilterOnnx {
namespace {

constexpr std::array<const char *, 4> kInputNames = {
    "speech",
    "analysis_cache",
    "synthesis_cache",
    "state_in",
};
constexpr std::array<const char *, 4> kOutputNames = {
    "enhanced",
    "analysis_cache_out",
    "synthesis_cache_out",
    "state_out",
};

} // namespace

bool is_dfsmn_ans_psm_48k_denoise(const std::vector<const char *> &input_names,
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

DfsmnAnsPsm48kDenoiseModel::DfsmnAnsPsm48kDenoiseModel(const DfsmnAnsPsm48kDenoiseModel &other,
                                                       const DenoiseConfig &config)
    : config_(config), state_size_(other.state_size_), hop_size_(other.hop_size_),
      sample_rate_(other.sample_rate_), delay_hops_(other.delay_hops_) {
    session_ = other.session_;
    input_names_ = other.input_names_;
    output_names_ = other.output_names_;
    reset();
}

std::unique_ptr<DenoiseModel> DfsmnAnsPsm48kDenoiseModel::init(const DenoiseConfig &config) {
    if (config.sample_rate != sample_rate_) {
        throw std::invalid_argument("DFSMN-ANS-PSM 48k model only supports a " +
                                    std::to_string(sample_rate_) + " Hz sample rate");
    }
    return std::unique_ptr<DenoiseModel>(new DfsmnAnsPsm48kDenoiseModel(*this, config));
}

void DfsmnAnsPsm48kDenoiseModel::reset() {
    state_.assign(state_size_, 0.0F);
    analysis_cache_.assign(hop_size_, 0.0F);
    synthesis_cache_.assign(hop_size_, 0.0F);
    input_buffer_.clear();
    input_offset_ = 0;
    received_samples_ = 0;
    emitted_samples_ = 0;
    pending_drop_ = delay_hops_;
    primed_ = false;
    finished_ = false;
}

std::vector<float> DfsmnAnsPsm48kDenoiseModel::forward(const float *speech) {
    const std::array<int64_t, 2> speech_shape = { 1, static_cast<int64_t>(hop_size_) };
    const std::array<int64_t, 1> state_shape = { static_cast<int64_t>(state_size_) };
    const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> inputs;
    inputs.reserve(4);
    inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, const_cast<float *>(speech),
                                                     hop_size_, speech_shape.data(),
                                                     speech_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, analysis_cache_.data(),
                                                     analysis_cache_.size(), speech_shape.data(),
                                                     speech_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, synthesis_cache_.data(),
                                                     synthesis_cache_.size(), speech_shape.data(),
                                                     speech_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(
        memory_info, state_.data(), state_.size(), state_shape.data(), state_shape.size()));

    auto outputs = session_->Run(Ort::RunOptions{ nullptr }, input_names_.data(), inputs.data(),
                                 inputs.size(), output_names_.data(), output_names_.size());
    std::vector<float> enhanced(hop_size_);
    std::copy_n(outputs[0].GetTensorData<float>(), hop_size_, enhanced.data());
    std::copy_n(outputs[1].GetTensorData<float>(), analysis_cache_.size(), analysis_cache_.data());
    std::copy_n(outputs[2].GetTensorData<float>(), synthesis_cache_.size(),
                synthesis_cache_.data());
    std::copy_n(outputs[3].GetTensorData<float>(), state_.size(), state_.data());
    return enhanced;
}

void DfsmnAnsPsm48kDenoiseModel::compact_input() {
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

std::vector<float> DfsmnAnsPsm48kDenoiseModel::decode(const float *data, int n,
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
    received_samples_ += static_cast<std::size_t>(n);
    std::vector<float> result;

    auto process_hop = [&](const float *hop) {
        auto enhanced = forward(hop);
        primed_ = true;
        if (pending_drop_ > 0) {
            --pending_drop_;
        } else {
            result.insert(result.end(), enhanced.begin(), enhanced.end());
        }
    };

    while (input_buffer_.size() - input_offset_ >= hop_size_) {
        process_hop(input_buffer_.data() + input_offset_);
        input_offset_ += hop_size_;
    }
    compact_input();

    if (input_finished) {
        if (!input_buffer_.empty()) {
            std::vector<float> padded(hop_size_, 0.0F);
            std::copy(input_buffer_.begin(), input_buffer_.end(), padded.begin());
            process_hop(padded.data());
        }
        // Drain the model's lookahead pipeline: `delay_hops_` extra
        // zero-input hops are needed to emit the last real hops of output,
        // regardless of whether the startup drop budget was already used
        // up during normal processing.
        if (primed_) {
            const std::vector<float> zeros(hop_size_, 0.0F);
            for (std::size_t i = 0; i < delay_hops_; ++i) {
                process_hop(zeros.data());
            }
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
