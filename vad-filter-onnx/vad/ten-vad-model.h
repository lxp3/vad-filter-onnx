#pragma once

#include "utils/pitch-estimator.h"
#include "vad/vad-model.h"
#include <memory>
#include <vector>

namespace VadFilterOnnx {

bool is_ten_vad(const std::vector<const char *> &input_names,
                const std::vector<const char *> &output_names);

class TenVadModel : public VadModel {
  public:
    TenVadModel() = default;
    TenVadModel(const VadModel &other, const VadConfig &config, int fs, int fl)
        : VadModel(other, config, fs, fl) {}

    std::unique_ptr<VadModel> init(const VadConfig &config) override;
    void init_state() override;
    float forward(float *data, int n) override;

  private:
    VadType type_ = VadType::TenVad;
    static constexpr std::array<int64_t, 2> state_shape_{ 1, 64 };
    static constexpr std::array<int64_t, 3> cache_shape_{ 1, 2, 41 };
    static constexpr std::array<int64_t, 2> pitch_shape_{ 1, 1 };

    // The 41st feature is an F0 estimate the ONNX graph cannot compute; see
    // utils/pitch-estimator.h.
    std::unique_ptr<PitchEstimator> pitch_estimator_;

    Ort::Value pitch_{ nullptr };
    Ort::Value h1_{ nullptr };
    Ort::Value c1_{ nullptr };
    Ort::Value h2_{ nullptr };
    Ort::Value c2_{ nullptr };
    Ort::Value conv_cache_{ nullptr };
};

} // namespace VadFilterOnnx
