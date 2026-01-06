#pragma once

#include "sliding-window-bit.h"
#include "vad-config.h"
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

namespace VadFilterOnnx {

class VadModel {
  public:
    // Factory method to load shared resources (Handle)
    static std::unique_ptr<VadModel> create(const std::string &path, int num_threads = 1,
                                            int device_id = -1);

    VadModel() = default;
    virtual ~VadModel() = default;

    // Create a new independent instance for inference sharing resources from this handle
    virtual std::unique_ptr<VadModel> init(const VadConfig &config) = 0;

    int get_sample_rate() const { return config_.sample_rate; }
    int get_frame_length_ms() const { return 1000 * frame_length_ / config_.sample_rate; }
    int get_frame_shift_ms() const { return 1000 * frame_shift_ / config_.sample_rate; }
    VadType get_vad_type() const { return type_; }

    virtual std::vector<VadSegment> decode(float *data, int n, bool input_finished);
    void flush();
    void reset();

  protected:
    // Protected constructor for sub-classes to share resources
    VadModel(const VadModel &other);

    virtual float forward(float *data, int n) = 0;
    virtual void init_state() = 0;
    void update_frame_state(float prob);
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
    int current_ = 0;
    int last_end_ = 0;
    int seg_idx_ = 0;
    std::vector<VadSegment> segs_;
    std::vector<float> reminder_;
};

} // namespace VadFilterOnnx
