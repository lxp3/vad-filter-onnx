#pragma once

#include "vad/vad-model.h"
extern "C" {
#include <fvad.h>
}
#include <cstdint>
#include <memory>
#include <vector>

namespace VadFilterOnnx {

void ValidateWebrtcVadConfig(const VadConfig &config);
int WebrtcVadFrameSamples(const VadConfig &config);

class WebrtcVadModel : public VadModel {
  public:
    WebrtcVadModel() = default;
    WebrtcVadModel(const VadModel &other, const VadConfig &config, int fs, int fl)
        : VadModel(other, config, fs, fl) {}

    std::unique_ptr<VadModel> init(const VadConfig &config) override;
    void init_state() override;
    float forward(float *data, int n) override;

  private:
    struct FvadDeleter {
        void operator()(Fvad *inst) const {
            if (inst != nullptr) {
                fvad_free(inst);
            }
        }
    };

    std::unique_ptr<Fvad, FvadDeleter> engine_;
    std::vector<int16_t> pcm16_;
};

} // namespace VadFilterOnnx
