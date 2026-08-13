#pragma once

#include "vad/vad-model.h"
#include <vector>

namespace VadFilterOnnx {

bool is_nemo_marblenet_vad(const std::vector<const char *> &input_names,
                           const std::vector<const char *> &output_names);

// NeMo MarbleNet v2.0 has no recurrent cache/state: every ONNX Runtime call is
// an independent forward pass over whatever waveform window is supplied, but
// the non-causal conv stack has a receptive field far wider than one 20ms
// output frame. Streaming is approximated the same way NeMo's own
// Online_Offline_Microphone_VAD_Demo does it: repeatedly run the full model
// over a sliding window that includes left/right context, keep only the
// "stable" frames in the middle whose receptive field is fully covered by
// real (non-edge) samples, then slide the window forward by the confirmed
// shift. This means a fixed extra latency of about right_context_samples_
// beyond FSMN/FireRedVAD-style true low-latency cached streaming.
class NemoMarbleNetVadModel : public VadModel {
  public:
    NemoMarbleNetVadModel() = default;
    NemoMarbleNetVadModel(const VadModel &other, const VadConfig &config, int fs, int fl)
        : VadModel(other, config, fs, fl) {}

    std::unique_ptr<VadModel> init(const VadConfig &config) override;
    void init_state() override;
    float forward(float *data, int n) override { return 0.0f; }
    std::vector<VadSegment> decode(float *data, int n, bool input_finished) override;

  private:
    std::vector<float> forward_window(const float *data, int n);
    void emit_frames(const std::vector<float> &probs, size_t frame_start, size_t frame_count);

    // Output frame granularity: encoder's stride-2 first block over 10ms mel
    // hops yields one probability every 20ms == 320 samples @16kHz.
    static constexpr int kOutputFrameSamples = 320;
    // Receptive field measured via gradient/impulse-response probing of the
    // exported encoder: a middle output frame's support spans roughly +-145
    // mel frames (+-1.45s); 1.5s of context on each side is a safe margin.
    static constexpr size_t kContextSamples = 24000; // 1.5 s
    // New samples confirmed per inference call: 300 ms == 15 output frames.
    static constexpr size_t kChunkShiftSamples = 4800;
    // Minimum samples the exported ONNX graph can produce a frame for.
    static constexpr size_t kMinWindowSamples = 400;

    std::vector<float> buffer_;
    // Total samples ever erased from the front of buffer_; buffer_[i]
    // corresponds to absolute sample index (dropped_total_ + i).
    size_t dropped_total_ = 0;
};

} // namespace VadFilterOnnx
