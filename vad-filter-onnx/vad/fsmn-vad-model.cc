#include "vad/fsmn-vad-model.h"
#include "utils/onnx-common.h"
#include <algorithm>
#include <string_view>

namespace VadFilterOnnx {

bool is_fsmn_vad(const std::vector<const char *> &input_names,
                 const std::vector<const char *> &output_names) {
    if (input_names.size() == 6 && output_names.size() == 5 &&
        std::string_view(input_names[0]) == "speech" &&
        std::string_view(input_names[1]) == "in_cache0" &&
        std::string_view(input_names[2]) == "in_cache1" &&
        std::string_view(input_names[3]) == "in_cache2" &&
        std::string_view(input_names[4]) == "in_cache3" &&
        std::string_view(input_names[5]) == "padding" &&
        std::string_view(output_names[0]) == "logits") {
        return true;
    }
    return false;
}

std::unique_ptr<VadModel> FsmnVadModel::init(const VadConfig &config) {
    int samples_per_ms = config.sample_rate / 1000;
    int frame_shift = 10 * samples_per_ms;
    int frame_length = 25 * samples_per_ms;
    auto instance = std::make_unique<FsmnVadModel>(*this, config, frame_shift, frame_length);
    instance->reset();
    return instance;
}

void FsmnVadModel::init_state() {
    is_first_inference_ = true;
    reminder_.clear();
    reminder_offset_ = 0;

    if (caches_.empty()) {
        // Initialize caches on first use
        for (int i = 0; i < 4; ++i) {
            caches_.emplace_back(Ort::Value::CreateTensor<float>(allocator_, cache_shape_.data(),
                                                                 cache_shape_.size()));
        }
    }

    for (int i = 0; i < 4; ++i) {
        Fill<float>(&caches_[i], 0.0f);
    }
}

std::vector<float> FsmnVadModel::forward_frames(float *data, int n, int32_t first_p,
                                                int32_t last_p) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    std::array<int64_t, 2> speech_shape = { 1, n };
    Ort::Value speech =
        Ort::Value::CreateTensor(memory_info, data, n, speech_shape.data(), speech_shape.size());

    std::array<int32_t, 2> padding_values = { first_p, last_p };
    std::array<int64_t, 1> padding_shape = { 2 };
    Ort::Value padding =
        Ort::Value::CreateTensor<int32_t>(memory_info, padding_values.data(), padding_values.size(),
                                          padding_shape.data(), padding_shape.size());

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(speech));
    for (int i = 0; i < 4; ++i) {
        inputs.push_back(std::move(caches_[i]));
    }
    inputs.push_back(std::move(padding));

    auto out = session_->Run(Ort::RunOptions{ nullptr }, input_names_.data(), inputs.data(),
                             inputs.size(), output_names_.data(), output_names_.size());

    // Update internal caches for the next streaming chunk
    for (int i = 0; i < 4; ++i) {
        caches_[i] = std::move(out[i + 1]);
    }

    // Extract logits from output tensor [1, T]
    float *logits_ptr = out[0].GetTensorMutableData<float>();
    auto shape = out[0].GetTensorTypeAndShapeInfo().GetShape();
    int T = static_cast<int>(shape[1]);

    // logits is noise probability
    // FunASR:  -1 < 1 - 2 * p_noise < 1
    // Ours: 0 < 2 * p_noise - 1 < 1 for speech probability, where p_speech = 1 - p_noise
    std::vector<float> speech_probs(T);
    for (int i = 0; i < T; ++i) {
        speech_probs[i] = 1 - logits_ptr[i];
    }

    return speech_probs;
}

void FsmnVadModel::process_logits(const std::vector<float> &logits) {
    for (int i = 0; i < logits.size(); ++i) {
        float p = logits[i];
        update_frame_state(p);
        current_ += frame_shift_;

        if (start_ != -1) {
            if (current_ - start_ > max_speech_samples_) {
                on_voice_end(current_);
                on_voice_start();
            }
        }
    }
}

std::vector<VadSegment> FsmnVadModel::decode(float *data, int n, bool input_finished) {
    received_samples_ += n;

    // Bound each ONNX invocation to roughly 100 ms. Four FBank frames
    // (55 ms) remain at the buffer front between invocations for LFR context.
    constexpr int kMaxNewFrames = 10;
    const size_t max_inference_samples =
        static_cast<size_t>(kMaxNewFrames - 1) * frame_shift_ + frame_length_;
    size_t input_offset = 0;

    while (input_offset < static_cast<size_t>(n) ||
           (input_finished && reminder_.size() > reminder_offset_)) {
        if (reminder_offset_ > 0 && (reminder_offset_ >= max_inference_samples ||
                                     reminder_offset_ * 2 >= reminder_.size())) {
            reminder_.erase(reminder_.begin(), reminder_.begin() + reminder_offset_);
            reminder_offset_ = 0;
        }

        size_t available = reminder_.size() - reminder_offset_;
        size_t room = max_inference_samples - std::min(available, max_inference_samples);
        size_t append_count = std::min(room, static_cast<size_t>(n) - input_offset);
        if (append_count > 0) {
            reminder_.insert(reminder_.end(), data + input_offset,
                             data + input_offset + append_count);
        }
        input_offset += append_count;
        available += append_count;

        const bool all_input_buffered = input_offset == static_cast<size_t>(n);
        const bool final_block = input_finished && all_input_buffered;
        const int min_frames = is_first_inference_ ? 3 : 5;
        const size_t min_samples =
            static_cast<size_t>(min_frames - 1) * frame_shift_ + frame_length_;

        if (final_block) {
            const size_t final_min_samples =
                is_first_inference_ ? static_cast<size_t>(frame_length_)
                                    : static_cast<size_t>(3 * frame_shift_ + frame_length_);
            if (available >= final_min_samples) {
                int32_t first_padding = is_first_inference_ ? 2 : 0;
                auto logits = forward_frames(reminder_.data() + reminder_offset_,
                                             static_cast<int>(available), first_padding, 2);
                process_logits(logits);
                is_first_inference_ = false;
            }
            reminder_.clear();
            reminder_offset_ = 0;
            break;
        }

        if (available < min_samples || (available < max_inference_samples && all_input_buffered)) {
            break;
        }

        // First block maps to [2, 0], middle blocks to [0, 0]. The first
        // block emits two left-padded logits, so it consumes two fewer shifts.
        int32_t first_padding = is_first_inference_ ? 2 : 0;
        auto logits = forward_frames(reminder_.data() + reminder_offset_,
                                     static_cast<int>(available), first_padding, 0);
        process_logits(logits);
        size_t consumed_frames = logits.size() - (is_first_inference_ ? 2 : 0);
        reminder_offset_ += consumed_frames * frame_shift_;
        is_first_inference_ = false;
    }

    if (input_finished) {
        flush();
        reminder_.clear();
        reminder_offset_ = 0;
    }

    std::vector<VadSegment> result = std::move(segs_);
    segs_.clear();
    return result;
}
} // namespace VadFilterOnnx
