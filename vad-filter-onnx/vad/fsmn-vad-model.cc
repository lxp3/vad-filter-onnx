#include "vad/fsmn-vad-model.h"
#include "utils/onnx-common.h"
#include <algorithm>
#include <string_view>

namespace VadFilterOnnx {

bool is_fsmn_vad(const std::vector<const char *> &input_names,
                 const std::vector<const char *> &output_names) {
    if (input_names.size() == 7 && output_names.size() == 5 &&
        std::string_view(input_names[0]) == "speech" &&
        std::string_view(input_names[1]) == "in_cache0" &&
        std::string_view(input_names[2]) == "in_cache1" &&
        std::string_view(input_names[3]) == "in_cache2" &&
        std::string_view(input_names[4]) == "in_cache3" &&
        std::string_view(input_names[5]) == "first_padding" &&
        std::string_view(input_names[6]) == "last_padding" &&
        std::string_view(output_names[0]) == "logits") {
        return true;
    }
    return false;
}

std::unique_ptr<VadModel> FsmnVadModel::init(const VadConfig &config) {
    auto instance = std::make_unique<FsmnVadModel>(static_cast<const VadModel &>(*this));
    instance->config_ = config;
    instance->type_ = VadType::FsmnVad;

    // FSMN VAD underlying frame configuration: 25ms frame length, 10ms frame shift
    instance->frame_shift_ = frame_shift_ms_ * (config.sample_rate / 1000);
    instance->frame_length_ = frame_length_ms_ * (config.sample_rate / 1000);

    // Window detector operates on the 10ms frame shift.
    // The buffer size must be at least as large as the largest window.
    int fs_ms = 10;
    int max_win_ms = std::max(config.speech_window_size_ms, config.silence_window_size_ms);
    int window_size = (max_win_ms + fs_ms - 1) / fs_ms;
    instance->window_detector_ = std::make_unique<SlidingWindowBit>(window_size);

    instance->reset();
    return instance;
}

void FsmnVadModel::init_state() {
    is_first_inference_ = true;
    caches_.clear();
    reminder_.clear(); // Ensure reminder buffer is cleared on state reset
    for (int i = 0; i < 4; ++i) {
        caches_.emplace_back(
            Ort::Value::CreateTensor<float>(allocator_, cache_shape_.data(), cache_shape_.size()));
        Fill<float>(&caches_.back(), 0.0f);
    }
}

std::vector<float> FsmnVadModel::forward_frames(float *data, int n, int64_t first_p,
                                                int64_t last_p) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    std::array<int64_t, 2> speech_shape = { 1, n };
    Ort::Value speech =
        Ort::Value::CreateTensor(memory_info, data, n, speech_shape.data(), speech_shape.size());

    std::array<int64_t, 1> p_shape = { 1 };
    // Padding parameters are passed as 0-dimensional tensors (scalars)
    Ort::Value first_padding =
        Ort::Value::CreateTensor<int64_t>(memory_info, &first_p, 1, p_shape.data(), 0);
    Ort::Value last_padding =
        Ort::Value::CreateTensor<int64_t>(memory_info, &last_p, 1, p_shape.data(), 0);

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(speech));
    for (int i = 0; i < 4; ++i) {
        inputs.push_back(std::move(caches_[i]));
    }
    inputs.push_back(std::move(first_padding));
    inputs.push_back(std::move(last_padding));

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

void FsmnVadModel::process_logits(const std::vector<float> &logits, int limit) {
    int n = (limit == -1) ? static_cast<int>(logits.size()) : limit;
    for (int i = 0; i < n; ++i) {
        float p = logits[i];
        update_frame_state(p);
        current_ += frame_shift_;

        if (start_ != -1) {
            int max_samples = (config_.max_speech_ms * config_.sample_rate) / 1000;
            if (current_ - start_ > max_samples) {
                on_voice_end();
                on_voice_start();
            }
        }
    }
}

std::vector<VadSegment> FsmnVadModel::decode(float *data, int n, bool input_finished) {
    // 1. Accumulate all new data into reminder buffer to ensure no data loss
    if (n > 0) {
        reminder_.insert(reminder_.end(), data, data + n);
    }

    // If no data is available and we're not finishing, wait for more data
    if (reminder_.empty() && !input_finished) {
        return {};
    }

    /*
     * FSMN-VAD LFR (Low Frame Rate) Streaming Logic:
     * - Frame Length: 25ms, Frame Shift: 10ms (160 samples @ 16kHz)
     * - LFR Layer: Concatenates 5 frames, Output Size = Input Size - 4.
     *
     * Streaming strategy (Preserving Context):
     * 1. First Inference (is_first_inference_):
     *    Wait for at least 100ms (1600 samples) of audio.
     *    Inference with first_padding=2.
     *    To maintain a 55ms (4 frames) context for the next step, we consume (N_real - 4) frames.
     *
     * 2. Normal Inference:
     *    Maintain a 55ms (880 samples) reminder to provide context for 10ms shift.
     *    Samples for N frames = (N-1)*10 + 25.
     *    A 55ms reminder contains 4 frames: (4-1)*10 + 25 = 55ms.
     *    After inference, we consume all produced scores and keep exactly 55ms + partial samples.
     *
     * 3. Alignment:
     *    Each score produced by the model represents 10ms of audio.
     *    'current_' is advanced by logits.size() * 10ms.
     */

    int fs = frame_shift_;                                      // 160 samples
    int fl = frame_length_;                                     // 400 samples
    int reminder_limit = 3 * fs + fl;                           // 55ms = 880 samples
    int first_chunk_limit = 100 * (config_.sample_rate / 1000); // 100ms = 1600 samples

    // 2. Process First Chunk or Normal Steady State
    if (is_first_inference_) {
        // Explicitly wait for enough data before first calculation to satisfy context requirements
        if (reminder_.size() < static_cast<size_t>(first_chunk_limit) && !input_finished) {
            return {};
        }

        int64_t first_p = 2;
        int64_t last_p = input_finished ? 2 : 0;
        auto logits =
            forward_frames(reminder_.data(), static_cast<int>(reminder_.size()), first_p, last_p);
        is_first_inference_ = false;

        if (input_finished) {
            // Process all results if audio ends here
            process_logits(logits);
            flush();
            reminder_.clear();
        } else {
            // Consume N_real - 4 frames to leave 55ms context.
            // logits.size() = N_real + 2 - 4 = N_real - 2.
            // num_to_consume = logits.size() - 2.
            int num_to_consume = std::max(0, static_cast<int>(logits.size()) - 2);
            process_logits(logits, num_to_consume);
            // Erase only consumed samples; all others remain in reminder_
            reminder_.erase(reminder_.begin(), reminder_.begin() + (num_to_consume * fs));
        }
    } else if (!input_finished) {
        // Normal state: process any new data beyond the 55ms reminder context
        if (reminder_.size() > static_cast<size_t>(reminder_limit)) {
            auto logits =
                forward_frames(reminder_.data(), static_cast<int>(reminder_.size()), 0, 0);

            // Consume all produced scores (N_real - 4), leaving the required 55ms context
            process_logits(logits);
            // Precise erasure: keeps exactly the last 4 frames + any sub-frame remainder
            reminder_.erase(reminder_.begin(), reminder_.begin() + (logits.size() * fs));
        }
    } else {
        // Final flush when input_finished = true
        if (!reminder_.empty()) {
            auto logits =
                forward_frames(reminder_.data(), static_cast<int>(reminder_.size()), 0, 2);
            process_logits(logits);
        }
        flush();
        reminder_.clear();
    }

    std::vector<VadSegment> result = std::move(segs_);
    segs_.clear();
    return result;
}
} // namespace VadFilterOnnx
