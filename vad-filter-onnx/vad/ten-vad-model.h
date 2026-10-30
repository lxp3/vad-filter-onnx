#pragma once

#include "utils/onnx-common.h"
#include "vad/vad-engine.h"
#include <memory>
#include <vector>

namespace VadFilterOnnx {

bool is_ten_vad(const std::vector<const char *> &input_names,
                const std::vector<const char *> &output_names);

std::unique_ptr<VadEngine> make_ten_engine(std::shared_ptr<const OnnxModel> model,
                                           const VadConfig &config);

} // namespace VadFilterOnnx
