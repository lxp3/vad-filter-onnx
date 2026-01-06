#pragma once

#include "vad/vad-model.h"
#include <vector>

namespace VadFilterOnnx {

bool is_fsmn_vad(const std::vector<const char *> &input_names,
                 const std::vector<const char *> &output_names);

class FsmnVadModel : public VadModel {
  public:
    FsmnVadModel() = default;
    explicit FsmnVadModel(const VadModel &other) : VadModel(other) {}

    std::unique_ptr<VadModel> init(const VadConfig &config) override;
    void init_state() override;
    float forward(float *data, int n) override { return 0.0f; };
    std::vector<VadSegment> decode(float *data, int n, bool input_finished) override;

  private:
    void process_logits(const std::vector<float> &logits, int limit = -1);
    std::vector<float> forward_frames(float *data, int n, int64_t first_p, int64_t last_p);
    std::vector<Ort::Value> caches_;
    // shape: [1, 128, 19, 1]
    static constexpr std::array<int64_t, 4> cache_shape_{ 1, 128, 19, 1 };
    static constexpr int frame_shift_ms_ = 10;
    static constexpr int frame_length_ms_ = 25;
    bool is_first_inference_ = true;
};

} // namespace VadFilterOnnx