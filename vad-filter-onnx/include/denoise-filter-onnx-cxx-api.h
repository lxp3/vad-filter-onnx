#pragma once

#include "denoise-config.h"
#include <memory>
#include <string>
#include <vector>

namespace VadFilterOnnx {

class AutoDenoiseModel {
  public:
    static std::unique_ptr<AutoDenoiseModel> create(const std::string &path, int num_threads = 1,
                                                    int device_id = -1);

    std::unique_ptr<AutoDenoiseModel> init(const DenoiseConfig &config);
    std::vector<float> decode(const float *data, int n, bool input_finished);
    void reset();

    ~AutoDenoiseModel();

  private:
    AutoDenoiseModel();
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace VadFilterOnnx
