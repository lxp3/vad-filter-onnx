#pragma once

#include "utils/onnx-common.h"
#include "vad/vad-engine.h"
#include <memory>
#include <vector>

namespace VadFilterOnnx {

bool is_nemo_marblenet_vad(const std::vector<const char *> &input_names,
                           const std::vector<const char *> &output_names);

// NeMo MarbleNet v2.0 has no recurrent cache/state: every ONNX Runtime call is an independent
// forward pass over whatever waveform window is supplied, but the non-causal conv stack has a
// receptive field far wider than one 20ms output frame. Streaming is approximated the same way
// NeMo's own Online_Offline_Microphone_VAD_Demo does it: repeatedly run the full model over a
// sliding window that includes left/right context, keep only the "stable" frames in the middle
// whose receptive field is fully covered by real (non-edge) samples, then slide the window
// forward by the confirmed shift. This means a fixed extra latency of about the right context
// beyond FSMN/FireRedVAD-style true low-latency cached streaming. 16 kHz only.
std::unique_ptr<VadEngine> make_nemo_marblenet_engine(std::shared_ptr<const OnnxModel> model,
                                                      const VadConfig &config);

} // namespace VadFilterOnnx
