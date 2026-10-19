#pragma once

#include "denoise/denoise-model.h"
#include <cstddef>

namespace VadFilterOnnx {

bool is_gtcrn_denoise(const std::vector<const char *> &input_names,
                      const std::vector<const char *> &output_names);

class GtcrnDenoiseModel : public DenoiseModel {
  public:
    GtcrnDenoiseModel() = default;
    std::unique_ptr<DenoiseModel> init(const DenoiseConfig &config) override;
    std::vector<float> decode(const float *data, int n, bool input_finished) override;
    void reset() override;

  private:
    GtcrnDenoiseModel(const GtcrnDenoiseModel &other, const DenoiseConfig &config);
    std::vector<float> forward(const float *speech);
    void compact_input();

    DenoiseConfig config_;
    std::vector<float> conv_cache_;
    std::vector<float> tra_cache_;
    std::vector<float> inter_cache_;
    std::vector<float> analysis_cache_;
    std::vector<float> synthesis_cache_;
    std::vector<float> input_buffer_;
    std::size_t input_offset_ = 0;
    std::size_t received_samples_ = 0;
    std::size_t emitted_samples_ = 0;
    bool primed_ = false;
    bool finished_ = false;
};

} // namespace VadFilterOnnx
