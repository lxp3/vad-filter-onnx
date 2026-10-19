#pragma once

#include "denoise-config.h"
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

namespace VadFilterOnnx {

class DenoiseModel {
  public:
    static std::unique_ptr<DenoiseModel> create(const std::string &path, int num_threads = 1,
                                                int device_id = -1);

    virtual ~DenoiseModel() = default;
    virtual std::unique_ptr<DenoiseModel> init(const DenoiseConfig &config) = 0;
    virtual std::vector<float> decode(const float *data, int n, bool input_finished) = 0;
    virtual void reset() = 0;

  protected:
    std::shared_ptr<Ort::Session> session_;
    std::vector<const char *> input_names_;
    std::vector<const char *> output_names_;
};

} // namespace VadFilterOnnx
