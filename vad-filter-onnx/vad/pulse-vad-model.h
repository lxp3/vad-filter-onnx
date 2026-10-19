#pragma once

#include "vad/vad-model.h"

namespace VadFilterOnnx {

bool is_pulse_vad(Ort::Session *session);

// PulseVAD has no recurrent cache/state. Each ONNX Runtime call consumes a
// fixed 200 ms waveform window and returns one speech probability.
class PulseVadModel : public VadModel {
  public:
    PulseVadModel() = default;
    PulseVadModel(const VadModel &other, const VadConfig &config, int fs, int fl)
        : VadModel(other, config, fs, fl) {}

    std::unique_ptr<VadModel> init(const VadConfig &config) override;
    void init_state() override;
    float forward(float *data, int n) override;
};

} // namespace VadFilterOnnx
