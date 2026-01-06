#pragma once

#include "vad/vad-model.h"
#include <vector>

bool is_silero_vad_v4(const std::vector<const char *> &input_names,
                      const std::vector<const char *> &output_names);
bool is_silero_vad_v5(const std::vector<const char *> &input_names,
                      const std::vector<const char *> &output_names);

class SileroVadModelV4 : public VadModel {
  public:
    SileroVadModelV4() = default;
    SileroVadModelV4(const VadModel &other, const VadConfig &config);

    void init_state() override;
    float forward(float *data, int n) override;

  private:
    static inline std::array<int64_t, 3> shape_{ 2, 1, 64 };
    Ort::Value h_state_{ nullptr };
    Ort::Value c_state_{ nullptr };
};

class SileroVadModelV5 : public VadModel {
  public:
    SileroVadModelV5() = default;
    SileroVadModelV5(const VadModel &other, const VadConfig &config);

    void init_state() override;
    float forward(float *data, int n) override;

  private:
    static inline std::array<int64_t, 3> shape_{ 2, 1, 128 };
    Ort::Value state_{ nullptr };
};
