#pragma once

#include "denoise/denoise-model.h"
#include <cstddef>

namespace VadFilterOnnx {

bool is_dpdfnet_denoise(const std::vector<const char *> &input_names,
                        const std::vector<const char *> &output_names);

class DpdfnetDenoiseModel : public DenoiseModel {
  public:
    DpdfnetDenoiseModel() = default;
    std::unique_ptr<DenoiseModel> init(const DenoiseConfig &config) override;
    std::vector<float> decode(const float *data, int n, bool input_finished) override;
    void reset() override;

    void set_state_size(std::size_t state_size) { state_size_ = state_size; }
    void set_hop_size(std::size_t hop_size) { hop_size_ = hop_size; }
    void set_sample_rate(int sample_rate) { sample_rate_ = sample_rate; }

  private:
    DpdfnetDenoiseModel(const DpdfnetDenoiseModel &other, const DenoiseConfig &config);
    std::vector<float> forward(const float *speech);
    void compact_input();

    DenoiseConfig config_;
    std::size_t state_size_ = 0;
    std::size_t hop_size_ = 0;
    int sample_rate_ = 0;
    std::vector<float> state_;
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
