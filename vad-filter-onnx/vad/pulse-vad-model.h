#pragma once

#include "utils/onnx-common.h"
#include "vad/vad-engine.h"
#include <memory>

namespace VadFilterOnnx {

bool is_pulse_vad(Ort::Session *session);

// PulseVAD has no recurrent cache/state. Each ONNX Runtime call consumes a fixed 200 ms waveform
// window and returns one speech probability.
std::unique_ptr<VadEngine> make_pulse_engine(std::shared_ptr<const OnnxModel> model,
                                             const VadConfig &config);

} // namespace VadFilterOnnx
