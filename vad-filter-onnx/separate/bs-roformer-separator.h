#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace VadFilterOnnx {

struct Separation {
    int sample_rate = 44100;
    std::vector<float> vocals;
    std::vector<float> instrumental;
};

class BsRoformerSeparator {
  public:
    explicit BsRoformerSeparator(const std::string &model_path, int num_threads = 1);
    ~BsRoformerSeparator();

    BsRoformerSeparator(const BsRoformerSeparator &) = delete;
    BsRoformerSeparator &operator=(const BsRoformerSeparator &) = delete;

    Separation Separate(const float *samples, int64_t frames, int channels, int sample_rate);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace VadFilterOnnx
