#pragma once

#include "denoise/denoise-model.h"
#include <cstddef>

namespace VadFilterOnnx {

bool is_mossformer2_se_48k_denoise(const std::vector<const char *> &input_names,
                                   const std::vector<const char *> &output_names);

// MossFormer2_SE_48K is a fully offline/non-causal denoise model: its
// generator (TestNet/MossFormer_MaskNet) has no cache/state I/O, and the
// ONNX graph takes and returns a raw waveform directly (Kaldi fbank
// frontend, STFT/masking/iSTFT baked into the graph). decode() buffers
// every sample it is given and never runs the ONNX session until
// input_finished == true.
//
// Mirrors MossformerganSe16kDenoiseModel's segmenting: inputs longer than
// one decode window (upstream's decode_window, 4 s at 48 kHz) are split
// into overlapping windows (75% stride), each run through the ONNX graph
// independently, and stitched back together by discarding the
// low-confidence edges of each segment (give_up_length on each side)
// before concatenating. Inputs shorter than one window run through the
// graph in a single call.
class Mossformer2Se48kDenoiseModel : public DenoiseModel {
  public:
    Mossformer2Se48kDenoiseModel() = default;
    std::unique_ptr<DenoiseModel> init(const DenoiseConfig &config) override;
    std::vector<float> decode(const float *data, int n, bool input_finished) override;
    void reset() override;

    void set_sample_rate(int sample_rate) { sample_rate_ = sample_rate; }

  private:
    Mossformer2Se48kDenoiseModel(const Mossformer2Se48kDenoiseModel &other,
                                 const DenoiseConfig &config);
    std::vector<float> forward(const float *data, std::size_t n);
    std::vector<float> decode_segmented();

    DenoiseConfig config_;
    int sample_rate_ = 0;
    std::vector<float> input_buffer_;
    bool finished_ = false;
};

} // namespace VadFilterOnnx
