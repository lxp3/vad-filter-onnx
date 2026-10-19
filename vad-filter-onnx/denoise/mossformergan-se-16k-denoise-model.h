#pragma once

#include "denoise/denoise-model.h"
#include <cstddef>

namespace VadFilterOnnx {

bool is_mossformergan_se_16k_denoise(const std::vector<const char *> &input_names,
                                     const std::vector<const char *> &output_names);

// MossFormerGAN_SE_16K is a fully offline/non-causal denoise model: its
// generator (SyncANet) has no cache/state I/O, and the ONNX graph takes
// and returns a raw waveform directly (STFT/power-compress/generator/
// power-uncompress/iSTFT baked into the graph). decode() buffers every
// sample it is given and never runs the ONNX session until
// input_finished == true.
//
// Unlike frcrn_se_16k.onnx (which runs the whole buffered utterance
// through the graph in one call), this backend mirrors upstream
// ClearerVoice-Studio's own decode_one_audio_mossformergan_se_16k
// segmenting algorithm: inputs longer than one decode window (10 s) are
// split into overlapping windows (75% stride), each run through the ONNX
// graph independently, and stitched back together by discarding the
// low-confidence edges of each segment (give_up_length on each side)
// before concatenating. Inputs shorter than one window run through the
// graph in a single call, matching upstream's non-segmented path.
class MossformerganSe16kDenoiseModel : public DenoiseModel {
  public:
    MossformerganSe16kDenoiseModel() = default;
    std::unique_ptr<DenoiseModel> init(const DenoiseConfig &config) override;
    std::vector<float> decode(const float *data, int n, bool input_finished) override;
    void reset() override;

    void set_sample_rate(int sample_rate) { sample_rate_ = sample_rate; }

  private:
    MossformerganSe16kDenoiseModel(const MossformerganSe16kDenoiseModel &other,
                                   const DenoiseConfig &config);
    std::vector<float> forward(const float *data, std::size_t n);
    std::vector<float> decode_segmented();

    DenoiseConfig config_;
    int sample_rate_ = 0;
    std::vector<float> input_buffer_;
    bool finished_ = false;
};

} // namespace VadFilterOnnx
