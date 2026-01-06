#pragma once

#include "sliding-window-bit.h"
#include "vad-config.h"
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

class VadModel {
  public:
    // Factory method to load model from disk and return a pointer to a specific implementation
    static VadModel *create(const std::string &path, int device_id = -1);

    VadModel() = default;
    // Constructor to share resources from another model instance
    VadModel(const VadModel &other, const VadConfig &config);

    int get_sample_rate() const { return config_.sample_rate; }
    int get_frame_length_ms() const { return 1000 * frame_length_ / config_.sample_rate; }
    int get_frame_shift_ms() const { return 1000 * frame_shift_ / config_.sample_rate; }
    VadType get_vad_type() const { return type_; }

    std::vector<VadSegment> decode(float *data, int n, bool input_finished);
    void flush();
    void reset();

    // Getters for resources
    const std::vector<const char *> &get_input_names() const { return input_names_; }
    const std::vector<const char *> &get_output_names() const { return output_names_; }
    std::shared_ptr<Ort::Session> get_session() const { return session_; }

  protected:
    virtual float forward(float *data, int n) = 0;
    virtual void init_state() = 0;
    void on_voice_start();
    void on_voice_end();

    VadType type_ = VadType::None;
    VadConfig config_;
    std::shared_ptr<Ort::Session> session_;
    std::vector<const char *> input_names_;
    std::vector<const char *> output_names_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::unique_ptr<SlidingWindowBit> window_detector_;

    // vad status
    int frame_length_ = 0;
    int frame_shift_ = 0;
    int start_ = -1; // Speech start position, -1 means silence
    int end_ = -1;   // Speech end position, -1 means not ended
    int current_sample_ = 0;
    int seg_idx_ = 0;
    std::vector<VadSegment> segs_;
    std::vector<float> reminder_;
};
