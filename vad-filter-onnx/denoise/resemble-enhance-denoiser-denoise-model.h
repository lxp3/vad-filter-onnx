#pragma once

#include "denoise/denoise-model.h"

namespace VadFilterOnnx {

bool is_resemble_enhance_denoiser_denoise(const std::vector<const char *> &input_names,
                                          const std::vector<const char *> &output_names);

// resemble-enhance's Denoiser (resemble_enhance.denoiser.denoiser.Denoiser)
// is a fully offline/non-causal denoise model: it has no cache/state I/O
// and its STFT/iSTFT + 2D-conv U-Net mask operate over the whole utterance,
// so it cannot be run incrementally. decode() buffers every sample it is
// given and never runs the ONNX session until input_finished == true; no
// output is ever returned before the stream finishes. Mirrors
// FrcrnSe16kDenoiseModel's single-shot (non-segmented) decode pattern.
class ResembleEnhanceDenoiserDenoiseModel : public DenoiseModel {
  public:
    ResembleEnhanceDenoiserDenoiseModel() = default;
    std::unique_ptr<DenoiseModel> init(const DenoiseConfig &config) override;
    std::vector<float> decode(const float *data, int n, bool input_finished) override;
    void reset() override;

    void set_sample_rate(int sample_rate) { sample_rate_ = sample_rate; }

  private:
    ResembleEnhanceDenoiserDenoiseModel(const ResembleEnhanceDenoiserDenoiseModel &other,
                                        const DenoiseConfig &config);
    std::vector<float> forward();

    DenoiseConfig config_;
    int sample_rate_ = 0;
    std::vector<float> input_buffer_;
    bool finished_ = false;
};

} // namespace VadFilterOnnx
