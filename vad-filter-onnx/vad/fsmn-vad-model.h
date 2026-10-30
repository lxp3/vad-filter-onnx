#pragma once

#include "utils/onnx-common.h"
#include "vad/vad-engine.h"
#include <memory>
#include <vector>

namespace VadFilterOnnx {

bool is_fsmn_vad(const std::vector<const char *> &input_names,
                 const std::vector<const char *> &output_names);

// FSMN-VAD (FunASR). Frame sizes follow the sample rate (10 ms shift, 25 ms length).
std::unique_ptr<VadEngine> make_fsmn_engine(std::shared_ptr<const OnnxModel> model,
                                            const VadConfig &config);

} // namespace VadFilterOnnx
