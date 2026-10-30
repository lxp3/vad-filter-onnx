#include "vad/silero-vad-model.h"
#include "utils/onnx-common.h"
#include <string_view>

namespace VadFilterOnnx {

bool is_silero_vad_v4(const std::vector<const char *> &input_names,
                      const std::vector<const char *> &output_names) {
    if (input_names.size() == 4 && output_names.size() == 3 &&
        std::string_view(input_names[0]) == "input" && std::string_view(input_names[1]) == "sr" &&
        std::string_view(input_names[2]) == "h" && std::string_view(input_names[3]) == "c" &&
        std::string_view(output_names[0]) == "output" &&
        std::string_view(output_names[1]) == "hn" && std::string_view(output_names[2]) == "cn") {
        return true;
    }
    return false;
}

bool is_silero_vad_v5(const std::vector<const char *> &input_names,
                      const std::vector<const char *> &output_names) {
    if (input_names.size() == 3 && output_names.size() == 2 &&
        std::string_view(input_names[0]) == "input" &&
        std::string_view(input_names[1]) == "state" &&
        std::string_view(output_names[0]) == "output" &&
        std::string_view(output_names[1]) == "stateN") {
        return true;
    }
    return false;
}

namespace {

// Silero V4: fixed 512-sample windows, LSTM state (h, c).
class SileroV4Engine final : public FrameEngine {
  public:
    SileroV4Engine(std::shared_ptr<const OnnxModel> model, const VadConfig &config)
        : FrameEngine(512, 512), ort_(std::move(model)), sample_rate_(config.sample_rate) {
        reset_state();
    }

  private:
    static constexpr std::array<int64_t, 3> kShape{ 2, 1, 64 };

    void reset_state() override {
        ResetState(h_state_, ort_.allocator(), kShape);
        ResetState(c_state_, ort_.allocator(), kShape);
    }

    float forward(float *data) override {
        int64_t sample_rate = sample_rate_;
        std::vector<Ort::Value> inputs;
        inputs.reserve(4);
        inputs.push_back(ort_.wave(data, frame_length_));
        inputs.push_back(ort_.scalar(&sample_rate));
        inputs.push_back(std::move(h_state_));
        inputs.push_back(std::move(c_state_));

        auto out = ort_.run(inputs);

        h_state_ = std::move(out[1]);
        c_state_ = std::move(out[2]);
        return out[0].GetTensorData<float>()[0];
    }

    OrtRunner ort_;
    int sample_rate_;
    Ort::Value h_state_{ nullptr };
    Ort::Value c_state_{ nullptr };
};

// Silero V5: shift 256/512 (8k/16k) plus 32/64 samples of context, single state tensor.
class SileroV5Engine final : public FrameEngine {
  public:
    SileroV5Engine(std::shared_ptr<const OnnxModel> model, const VadConfig &config)
        : FrameEngine(config.sample_rate == 8000 ? 256 : 512,
                      (config.sample_rate == 8000 ? 256 : 512) +
                          (config.sample_rate == 8000 ? 32 : 64)),
          ort_(std::move(model)),
          sample_rate_(config.sample_rate) {
        reset_state();
    }

  private:
    static constexpr std::array<int64_t, 3> kShape{ 2, 1, 128 };

    void reset_state() override { ResetState(state_, ort_.allocator(), kShape); }

    float forward(float *data) override {
        // Some exports also take the sample rate as a third input.
        int64_t sample_rate = sample_rate_;
        std::vector<Ort::Value> inputs;
        inputs.reserve(3);
        inputs.push_back(ort_.wave(data, frame_length_));
        inputs.push_back(std::move(state_));
        if (ort_.num_inputs() > 2)
            inputs.push_back(ort_.scalar(&sample_rate));

        auto out = ort_.run(inputs);

        state_ = std::move(out[1]);
        return out[0].GetTensorData<float>()[0];
    }

    OrtRunner ort_;
    int sample_rate_;
    Ort::Value state_{ nullptr };
};

} // namespace

std::unique_ptr<VadEngine> make_silero_v4_engine(std::shared_ptr<const OnnxModel> model,
                                                 const VadConfig &config) {
    return std::make_unique<SileroV4Engine>(std::move(model), config);
}

std::unique_ptr<VadEngine> make_silero_v5_engine(std::shared_ptr<const OnnxModel> model,
                                                 const VadConfig &config) {
    return std::make_unique<SileroV5Engine>(std::move(model), config);
}

} // namespace VadFilterOnnx
