#pragma once

#include "denoise/denoise-model.h"
#include <cstddef>

namespace VadFilterOnnx {

bool is_dfsmn_ans_psm_48k_denoise(const std::vector<const char *> &input_names,
                                  const std::vector<const char *> &output_names);

// DFSMN-ANS-PSM 48k causal streaming denoise model. Shares the same
// speech/analysis_cache/synthesis_cache/state_in -> enhanced/
// analysis_cache_out/synthesis_cache_out/state_out ONNX I/O contract as
// DpdfnetDenoiseModel/DeepFilterNetDenoiseModel; state_size_/hop_size_/
// sample_rate_/delay_hops_ are read from ONNX metadata.
class DfsmnAnsPsm48kDenoiseModel : public DenoiseModel {
  public:
    DfsmnAnsPsm48kDenoiseModel() = default;
    std::unique_ptr<DenoiseModel> init(const DenoiseConfig &config) override;
    std::vector<float> decode(const float *data, int n, bool input_finished) override;
    void reset() override;

    void set_state_size(std::size_t state_size) { state_size_ = state_size; }
    void set_hop_size(std::size_t hop_size) { hop_size_ = hop_size; }
    void set_sample_rate(int sample_rate) { sample_rate_ = sample_rate; }
    // Number of hops of pure algorithmic delay the model introduces (the
    // overlap-add analysis window straddles the previous hop and this
    // hop), i.e. how many leading hops of model output must be dropped and
    // how many trailing zero-hops must be flushed through at
    // `input_finished` to keep output sample count aligned with input
    // sample count.
    void set_delay_hops(std::size_t delay_hops) { delay_hops_ = delay_hops; }

  private:
    DfsmnAnsPsm48kDenoiseModel(const DfsmnAnsPsm48kDenoiseModel &other, const DenoiseConfig &config);
    std::vector<float> forward(const float *speech);
    void compact_input();

    DenoiseConfig config_;
    std::size_t state_size_ = 0;
    std::size_t hop_size_ = 0;
    int sample_rate_ = 0;
    std::size_t delay_hops_ = 1;
    std::vector<float> state_;
    std::vector<float> analysis_cache_;
    std::vector<float> synthesis_cache_;
    std::vector<float> input_buffer_;
    std::size_t input_offset_ = 0;
    std::size_t received_samples_ = 0;
    std::size_t emitted_samples_ = 0;
    // Startup: number of leading real hops still to be dropped.
    std::size_t pending_drop_ = 0;
    // Whether at least one hop has been processed (so a shutdown flush of
    // `delay_hops_` zero-hops is needed at input_finished to drain the
    // model's lookahead pipeline).
    bool primed_ = false;
    bool finished_ = false;
};

} // namespace VadFilterOnnx
