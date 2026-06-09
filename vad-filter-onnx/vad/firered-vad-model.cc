#include "vad/firered-vad-model.h"
#include "utils/onnx-common.h"
#include <algorithm>
#include <stdexcept>
#include <string_view>

namespace VadFilterOnnx {

bool is_firered_vad(const std::vector<const char *> &input_names,
                    const std::vector<const char *> &output_names) {
    return input_names.size() == 2 && output_names.size() == 2 &&
           std::string_view(input_names[0]) == "speech" &&
           std::string_view(input_names[1]) == "caches_in" &&
           std::string_view(output_names[0]) == "probs" &&
           std::string_view(output_names[1]) == "caches_out";
}

std::unique_ptr<VadModel> FireredVadModel::init(const VadConfig &config) {
    if (config.sample_rate != 16000) {
        throw std::runtime_error("FireRedVAD supports 16000 Hz audio only");
    }

    constexpr int frame_shift = 160;  // 10 ms at 16 kHz
    constexpr int frame_length = 400; // 25 ms at 16 kHz
    auto instance = std::make_unique<FireredVadModel>(*this, config, frame_shift, frame_length);
    instance->reset();
    return instance;
}

void FireredVadModel::init_state() {
    reminder_.clear();

    if (caches_ == nullptr) {
        caches_ = Ort::Value::CreateTensor<float>(allocator_, cache_shape_.data(),
                                                  cache_shape_.size());
    }
    Fill<float>(&caches_, 0.0f);
}

std::vector<float> FireredVadModel::forward_frames(float *data, int n) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    std::array<int64_t, 2> speech_shape = { 1, n };
    Ort::Value speech =
        Ort::Value::CreateTensor(memory_info, data, n, speech_shape.data(), speech_shape.size());

    std::vector<Ort::Value> inputs;
    inputs.reserve(2);
    inputs.push_back(std::move(speech));
    inputs.push_back(std::move(caches_));

    auto out = session_->Run(Ort::RunOptions{ nullptr }, input_names_.data(), inputs.data(),
                             inputs.size(), output_names_.data(), output_names_.size());

    caches_ = std::move(out[1]);

    const float *probs_ptr = out[0].GetTensorData<float>();
    auto shape = out[0].GetTensorTypeAndShapeInfo().GetShape();
    int T = static_cast<int>(shape[1]);

    std::vector<float> probs(T);
    for (int i = 0; i < T; ++i) {
        probs[i] = probs_ptr[i];
    }
    return probs;
}

void FireredVadModel::process_probs(const std::vector<float> &probs) {
    for (float prob : probs) {
        update_frame_state(prob);
        current_ += frame_shift_;

        if (start_ != -1 && current_ - start_ > max_speech_samples_) {
            on_voice_end(current_);
            on_voice_start();
        }
    }
}

std::vector<VadSegment> FireredVadModel::decode(float *data, int n, bool input_finished) {
    received_samples_ += n;
    if (n > 0) {
        reminder_.insert(reminder_.end(), data, data + n);
    }

    if (reminder_.size() >= static_cast<size_t>(frame_length_)) {
        int available = static_cast<int>(reminder_.size());
        int num_frames = (available - frame_length_) / frame_shift_ + 1;
        int num_samples = frame_length_ + (num_frames - 1) * frame_shift_;

        auto probs = forward_frames(reminder_.data(), num_samples);
        process_probs(probs);

        if (!input_finished) {
            int consume = static_cast<int>(probs.size()) * frame_shift_;
            consume = std::min(consume, static_cast<int>(reminder_.size()));
            reminder_.erase(reminder_.begin(), reminder_.begin() + consume);
        }
    } else if (!input_finished) {
        return {};
    }

    if (input_finished) {
        flush();
        reminder_.clear();
    }

    std::vector<VadSegment> result = std::move(segs_);
    segs_.clear();
    return result;
}

} // namespace VadFilterOnnx
