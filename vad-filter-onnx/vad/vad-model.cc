#include "vad/vad-model.h"
#include "utils/onnx-common.h"
#include "vad/fsmn-vad-model.h"
#include "vad/silero-vad-model.h"
// #include <format>
// #include <iostream>

std::unique_ptr<VadModel> VadModel::create(const std::string &path, int device_id) {
    std::shared_ptr<Ort::Session> session = ReadOnnx(path, 1, device_id);
    std::vector<const char *> input_names, output_names;
    GetInputOutputInfo(session, input_names, output_names);

    // Create a temporary resource holder to identify the model type
    std::unique_ptr<VadModel> model;
    if (is_silero_vad_v4(input_names, output_names)) {
        model = std::make_unique<SileroVadModelV4>();
        model->type_ = VadType::SileroVadV4;
        printf("Success to create SileroVadV4 model from %s\n", path.c_str());
    } else if (is_silero_vad_v5(input_names, output_names)) {
        model = std::make_unique<SileroVadModelV5>();
        model->type_ = VadType::SileroVadV5;
        printf("Success to create SileroVadV5 model from %s\n", path.c_str());
    } else if (is_fsmn_vad(input_names, output_names)) {
        model = std::make_unique<FsmnVadModel>();
        model->type_ = VadType::FsmnVad;
        printf("Success to create FsmnVad model from %s\n", path.c_str());
    } else {
        printf("ERROR: Unknown Vad model type in %s\n", path.c_str());
        return nullptr;
    }

    model->session_ = session;
    model->input_names_ = std::move(input_names);
    model->output_names_ = std::move(output_names);
    return model;
}

VadModel::VadModel(const VadModel &other)
    : type_(other.type_),
      session_(other.session_),
      input_names_(other.input_names_),
      output_names_(other.output_names_) {}

void VadModel::reset() {
    init_state();
    current_ = 0;
    last_end_ = 0;
    start_ = -1;
    end_ = -1;
    seg_idx_ = 0;
    segs_.clear();
}

void VadModel::on_voice_start() {
    // Basic start calculation with padding
    int padding_samples = (config_.left_padding_ms * config_.sample_rate) / 1000;
    int num_right_ones = window_detector_->num_right_ones();
    start_ = current_ - (num_right_ones * frame_shift_) - padding_samples;
    start_ = std::max(last_end_, start_);

    VadSegment seg;
    seg.idx = seg_idx_;
    seg.start = start_;
    seg.start_ms = (start_ * 1000) / config_.sample_rate;
    segs_.push_back(seg);
}

void VadModel::on_voice_end() {
    int padding_samples = (config_.right_padding_ms * config_.sample_rate) / 1000;
    int num_right_zeros = window_detector_->num_right_zeros();
    end_ = current_ - (num_right_zeros * frame_shift_) + padding_samples;
    end_ = std::min(end_, current_);

    // If on_voice_start was called in the same decode() call, segs_ already has a partial segment.
    if (!segs_.empty() && segs_.back().end == -1) {
        auto &last_seg = segs_.back();
        last_seg.end = end_;
        last_seg.end_ms = (end_ * 1000) / config_.sample_rate;
    } else {
        // Speech started in a previous decode() call, need to add the finished segment.
        segs_.emplace_back(seg_idx_, start_, end_, (start_ * 1000) / config_.sample_rate,
                           (end_ * 1000) / config_.sample_rate);
    }

    last_end_ = end_;
    start_ = -1;
    end_ = -1;
    seg_idx_++;
}

void VadModel::update_frame_state(float prob) {
    bool is_speech = prob > config_.threshold;
    window_detector_->push(is_speech);

    // std::cout << std::format(
    //     "current_ {:.3f} s | start_ {:.3f} s | end_ {:.3f} s | prob {:.3f} | is_speech {}\n",
    //     current_ * 1.0 / config_.sample_rate, start_ * 1.0 / config_.sample_rate,
    //     end_ * 1.0 / config_.sample_rate, prob, is_speech);

    if (start_ == -1) {
        if (window_detector_->is_up()) {
            on_voice_start();
        }
    } else {
        if (window_detector_->is_down()) {
            on_voice_end();
        }
    }
}

void VadModel::flush() {
    if (start_ != -1) {
        on_voice_end();
    }
}

std::vector<VadSegment> VadModel::decode(float *data, int n, bool input_finished) {
    if (n == 0 && !input_finished) {
        return {};
    }

    int overlap_length = frame_length_ - frame_shift_;

    // 1. Accumulate data if we have leftovers from previous call
    if (!reminder_.empty()) {
        reminder_.insert(reminder_.end(), data, data + n);
    }

    // Determine processing source: reminder buffer or direct input pointer
    const float *ptr = reminder_.empty() ? data : reminder_.data();
    int len = reminder_.empty() ? n : static_cast<int>(reminder_.size());

    // 2. Main inference loop: process frames by shifting window
    while (len >= frame_length_) {
        float prob = forward(const_cast<float *>(ptr), frame_length_);
        update_frame_state(prob);

        // Check if current speech segment exceeds maximum allowed duration
        if (start_ != -1) {
            int max_samples = (config_.max_speech_ms * config_.sample_rate) / 1000;
            if (current_ - start_ > max_samples) {
                on_voice_end();
                on_voice_start();
            }
        }

        // Advance pointers and counters by frame_shift_
        ptr += frame_shift_;
        len -= frame_shift_;
        current_ += frame_shift_;
    }

    // 3. Finalization or buffer state preservation
    if (input_finished) {
        // Force close any active speech segment at the end of input
        flush();
        reminder_.clear();
    } else {
        // Save unconsumed data and required overlap for the next decode call
        if (len > 0) {
            if (!reminder_.empty()) {
                std::vector<float> next_reminder(ptr, ptr + len);
                reminder_ = std::move(next_reminder);
            } else {
                reminder_.assign(ptr, ptr + len);
            }
        } else {
            reminder_.clear();
        }

        // For online/streaming: if speech is active but no boundary event
        // occurred in this call, report it as a partial segment.
        if (start_ != -1 && segs_.empty()) {
            segs_.emplace_back(seg_idx_, start_, -1, (start_ * 1000) / config_.sample_rate, -1);
        }
    }

    // Move collected segments to result and clear local cache
    std::vector<VadSegment> result_segments = std::move(segs_);
    segs_.clear();
    return result_segments;
}
