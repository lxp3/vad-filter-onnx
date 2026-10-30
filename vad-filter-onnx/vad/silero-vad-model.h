#pragma once

#include "utils/onnx-common.h"
#include "vad/vad-engine.h"
#include <memory>
#include <vector>

namespace VadFilterOnnx {

bool is_silero_vad_v4(const std::vector<const char *> &input_names,
                      const std::vector<const char *> &output_names);
bool is_silero_vad_v5(const std::vector<const char *> &input_names,
                      const std::vector<const char *> &output_names);

std::unique_ptr<VadEngine> make_silero_v4_engine(std::shared_ptr<const OnnxModel> model,
                                                 const VadConfig &config);
std::unique_ptr<VadEngine> make_silero_v5_engine(std::shared_ptr<const OnnxModel> model,
                                                 const VadConfig &config);

} // namespace VadFilterOnnx
