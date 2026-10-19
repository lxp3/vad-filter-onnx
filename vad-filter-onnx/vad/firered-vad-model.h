#pragma once

#include "vad/vad-model.h"
#include <array>
#include <vector>

namespace VadFilterOnnx {

bool is_firered_vad(const std::vector<const char *> &input_names,
                    const std::vector<const char *> &output_names);

class FireredVadModel : public VadModel {
  public:
    FireredVadModel() = default;
    FireredVadModel(const VadModel &other, const VadConfig &config, int fs, int fl)
        : VadModel(other, config, fs, fl) {}

    std::unique_ptr<VadModel> init(const VadConfig &config) override;
    void init_state() override;
    float forward(float *data, int n) override { return 0.0f; }
    std::vector<VadSegment> decode(float *data, int n, bool input_finished) override;

  private:
    void process_probs(const std::vector<float> &probs);
    std::vector<float> forward_frames(float *data, int n);

    Ort::Value caches_{ nullptr };
    static constexpr std::array<int64_t, 4> cache_shape_{ 8, 1, 128, 19 };
    size_t reminder_offset_ = 0;
};

} // namespace VadFilterOnnx
