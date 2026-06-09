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

    virtual std::vector<VadSegment> decode(float *data, int n, bool input_finished);
    void setup_config(const VadConfig &config);
    const VadConfig &get_config() const;
    VadSegment flush();
    void reset();

  protected:
    // Protected constructor for sub-classes to share resources and pre-calculate parameters
    VadModel(const VadModel &other, const VadConfig &config, int frame_shift, int frame_length);

    virtual float forward(float *data, int n) = 0;
    virtual void init_state() = 0;
    void apply_config(const VadConfig &config);
    void update_frame_state(float prob);
    void on_voice_start();
    void on_voice_end(int end_limit_samples);

    VadType type_ = VadType::None;
    VadConfig config_;
    bool configured_ = false;
    std::shared_ptr<Ort::Session> session_;
    std::vector<const char *> input_names_;
    std::vector<const char *> output_names_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::unique_ptr<SlidingWindowBit> window_detector_;

    // Pre-calculated parameters (in samples or frames)
    int samples_per_ms_;
    int frame_length_;
    int frame_shift_;
    int speech_window_size_frames_;
    int silence_window_size_frames_;
    int speech_window_threshold_frames_;
    int silence_window_threshold_frames_;
    int left_padding_samples_;
    int right_padding_samples_;
    int max_speech_samples_;

    // vad status
    int start_ = -1; // Speech start position, -1 means silence
    int end_ = -1;   // Speech end position, -1 means not ended
    int current_ = 0;
    int received_samples_ = 0;
    int last_end_ = 0;
    int seg_idx_ = 0;
    std::vector<VadSegment> segs_;
    std::vector<float> reminder_;
};

} // namespace VadFilterOnnx
