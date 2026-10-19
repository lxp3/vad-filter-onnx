#include "vad/nemo-marblenet-vad-model.h"
#include "utils/onnx-common.h"
#include <algorithm>
#include <stdexcept>
#include <string_view>

namespace VadFilterOnnx {

bool is_nemo_marblenet_vad(const std::vector<const char *> &input_names,
                           const std::vector<const char *> &output_names) {
    return input_names.size() == 1 && output_names.size() == 1 &&
           std::string_view(input_names[0]) == "speech" &&
           std::string_view(output_names[0]) == "probs";
}

std::unique_ptr<VadModel> NemoMarbleNetVadModel::init(const VadConfig &config) {
    if (config.sample_rate != 16000) {
        throw std::runtime_error("NeMo MarbleNet VAD supports 16000 Hz audio only");
    }

    // frame_shift_ is the output frame granularity (20 ms); frame_length_ is
    // unused since decode() is fully overridden below (no cache/state, but
    // wide non-causal receptive field means each output frame needs a whole
    // sliding window forward pass, not a single-frame forward()).
    auto instance = std::make_unique<NemoMarbleNetVadModel>(*this, config, kOutputFrameSamples, 0);
    instance->reset();
    return instance;
}

void NemoMarbleNetVadModel::init_state() {
    buffer_.clear();
    dropped_total_ = 0;
}

std::vector<float> NemoMarbleNetVadModel::forward_window(const float *data, int n) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    std::array<int64_t, 2> speech_shape = { 1, n };
    Ort::Value speech = Ort::Value::CreateTensor(memory_info, const_cast<float *>(data), n,
                                                 speech_shape.data(), speech_shape.size());

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(speech));

    auto out = session_->Run(Ort::RunOptions{ nullptr }, input_names_.data(), inputs.data(),
                             inputs.size(), output_names_.data(), output_names_.size());

    const float *probs_ptr = out[0].GetTensorData<float>();
    auto shape = out[0].GetTensorTypeAndShapeInfo().GetShape();
    int T = static_cast<int>(shape[1]);

    return std::vector<float>(probs_ptr, probs_ptr + T);
}

void NemoMarbleNetVadModel::emit_frames(const std::vector<float> &probs, size_t frame_start,
                                        size_t frame_count) {
    for (size_t i = 0; i < frame_count; ++i) {
        float prob = probs[frame_start + i];
        update_frame_state(prob);
        current_ += frame_shift_;

        if (start_ != -1 && current_ - start_ > max_speech_samples_) {
            on_voice_end(current_);
            on_voice_start();
        }
    }
}

std::vector<VadSegment> NemoMarbleNetVadModel::decode(float *data, int n, bool input_finished) {
    received_samples_ += n;

    // buffer_[0] always corresponds to absolute sample index
    // (current_ - kContextSamples), i.e. it retains exactly the left context
    // needed for the next window; trim anything older every iteration below.
    buffer_.insert(buffer_.end(), data, data + n);

    while (true) {
        // Samples in buffer_ beyond current_ that have not yet been
        // confirmed as output frames.
        size_t confirmed_offset = static_cast<size_t>(current_) - dropped_total_;
        size_t new_available = buffer_.size() - confirmed_offset;

        bool all_input_buffered = true; // buffer_ already holds everything appended so far
        bool final_block = input_finished && all_input_buffered;

        if (!final_block && new_available < kChunkShiftSamples) {
            break;
        }
        if (new_available == 0) {
            break;
        }

        size_t shift_samples = final_block ? new_available : kChunkShiftSamples;

        size_t left_context = std::min(confirmed_offset, kContextSamples);
        size_t window_start = confirmed_offset - left_context;
        size_t window_end = std::min(buffer_.size(), confirmed_offset + shift_samples + kContextSamples);
        size_t window_len = window_end - window_start;

        if (window_len < kMinWindowSamples) {
            if (!final_block) {
                break;
            }
            // Too little data left to produce even one frame; drop the tail.
            buffer_.clear();
            break;
        }

        auto probs = forward_window(buffer_.data() + window_start, static_cast<int>(window_len));

        size_t keep_start_frame = left_context / static_cast<size_t>(frame_shift_);
        size_t keep_count = final_block
            ? (probs.size() > keep_start_frame ? probs.size() - keep_start_frame : 0)
            : shift_samples / static_cast<size_t>(frame_shift_);
        keep_count = std::min(keep_count, probs.size() > keep_start_frame ? probs.size() - keep_start_frame : 0);

        emit_frames(probs, keep_start_frame, keep_count);

        // Drop everything before the new left-context start for the next window.
        size_t new_confirmed_offset = static_cast<size_t>(current_) - dropped_total_;
        size_t drop_count = new_confirmed_offset > kContextSamples ? new_confirmed_offset - kContextSamples : 0;
        if (drop_count > 0) {
            buffer_.erase(buffer_.begin(), buffer_.begin() + drop_count);
            dropped_total_ += drop_count;
        }

        if (final_block) {
            buffer_.clear();
            break;
        }
    }

    if (input_finished) {
        flush();
        buffer_.clear();
    }

    std::vector<VadSegment> result = std::move(segs_);
    segs_.clear();
    return result;
}

} // namespace VadFilterOnnx
