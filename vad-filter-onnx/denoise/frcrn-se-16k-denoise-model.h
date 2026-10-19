#pragma once

#include "denoise/denoise-model.h"

namespace VadFilterOnnx {

bool is_frcrn_se_16k_denoise(const std::vector<const char *> &input_names,
                      const std::vector<const char *> &output_names);

// FRCRN is a fully offline/non-causal denoise model: it has no cache/state
// I/O. decode() buffers incoming samples and runs a single ONNX forward
// only when input_finished == true. Each decode stream is limited to at
// most 1 second (sample_rate_ samples); longer audio must be chunked by
// the caller. No output is returned before the stream finishes.
class FrcrnSe16kDenoiseModel : public DenoiseModel {
  public:
    FrcrnSe16kDenoiseModel() = default;
    std::unique_ptr<DenoiseModel> init(const DenoiseConfig &config) override;
    std::vector<float> decode(const float *data, int n, bool input_finished) override;
    void reset() override;

    void set_sample_rate(int sample_rate) { sample_rate_ = sample_rate; }

  private:
    FrcrnSe16kDenoiseModel(const FrcrnSe16kDenoiseModel &other, const DenoiseConfig &config);
    std::vector<float> forward();

    DenoiseConfig config_;
    int sample_rate_ = 0;
    std::vector<float> input_buffer_;
    bool finished_ = false;
};

} // namespace VadFilterOnnx
