#include "vad/vad-model.h"
#include "utils/onnx-common.h"
#include "vad/silero-vad-model.h"

VadModel *VadModel::create(const std::string &path, int device_id) {
    std::shared_ptr<Ort::Session> session = ReadOnnx(path, 1, device_id);
    std::vector<const char *> input_names, output_names;
    GetInputOutputInfo(session, input_names, output_names);

    // Create a temporary resource holder to identify the model type
    VadModel *model = nullptr;
    if (is_silero_vad_v4(input_names, output_names)) {
        model = new SileroVadModelV4();
        model->type_ = VadType::SileroVadV4;
        printf("Success to create SileroVadV4 model from %s\n", path.c_str());
    } else if (is_silero_vad_v5(input_names, output_names)) {
        printf("Success to create SileroVadV5 model from %s\n", path.c_str());
        model = new SileroVadModelV5();
        model->type_ = VadType::SileroVadV5;
    } else {
        printf("ERROR: Unknown Vad model type in %s\n", path.c_str());
        return nullptr;
    }

    model->session_ = session;
    model->input_names_ = std::move(input_names);
    model->output_names_ = std::move(output_names);
    return model;
}

VadModel::VadModel(const VadModel &other, const VadConfig &config)
    : config_(config),
      session_(other.session_),
      input_names_(other.input_names_),
      output_names_(other.output_names_) {

    if (other.type_ == VadType::SileroVadV4) {
        frame_shift_ = 512;
        frame_length_ = 512;
    } else if (other.type_ == VadType::SileroVadV5) {
        frame_shift_ = (config.sample_rate == 8000 ? 256 : 512);
        int context_size = (config.sample_rate == 8000 ? 32 : 64);
        frame_length_ = frame_shift_ + context_size;
    } else {
        printf("ERROR: Unknown Vad model type\n");
        exit(-1);
    }

    // initialize sliding window detector
    int frame_shift_ms = get_frame_shift_ms();
    int window_size = (config_.window_size_ms + frame_shift_ms - 1) / frame_shift_ms;
    int window_threshold = (config_.min_speech_ms + frame_shift_ms - 1) / frame_shift_ms;
    window_detector_ = std::make_unique<SlidingWindowBit>(window_size, window_threshold);

    // reset model
    reset();
}

void VadModel::reset() {
    init_state();
    current_sample_ = 0;
    start_ = -1;
    end_ = -1;
    seg_idx_ = 0;
    segs_.clear();
}

void VadModel::on_voice_start() {
    // Basic start calculation with padding
    int padding_samples = (config_.left_padding_ms * config_.sample_rate) / 1000;
    start_ = std::max(0, current_sample_ - padding_samples);

    VadSegment seg;
    seg.idx = seg_idx_;
    seg.start = start_;
    seg.start_ms = (start_ * 1000) / config_.sample_rate;
    segs_.push_back(seg);
}

void VadModel::on_voice_end() {
    int padding_samples = (config_.right_padding_ms * config_.sample_rate) / 1000;
    end_ = current_sample_ + padding_samples;

    if (!segs_.empty()) {
        auto &last_seg = segs_.back();
        last_seg.end = end_;
        last_seg.end_ms = (end_ * 1000) / config_.sample_rate;
    }

    start_ = -1;
    end_ = -1;
    seg_idx_++;
}

void VadModel::flush() {}

std::vector<VadSegment> VadModel::decode(float *data, int n, bool input_finished) {
    std::vector<VadSegment> result_segments;

    while (buffer_.size() >= window_size_samples_) {
        float prob = forward(buffer_.data(), window_size_samples_);
        get_frame_state(prob);

        // Remove processed samples. Note: for Silero, we usually advance by window_shift_samples_
        buffer_.erase(buffer_.begin(), buffer_.begin() + window_shift_samples_);
        current_sample_ += window_shift_samples_;
    }

    if (input_finished && start_ != -1) {
        on_voice_end();
    }

    // Move detected segments to result
    if (!segments_.empty()) {
        result_segments = std::move(segments_);
        segments_.clear();
    }

    return result_segments;
}
