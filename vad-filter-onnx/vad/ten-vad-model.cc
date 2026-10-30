#include "vad/ten-vad-model.h"
#include "utils/onnx-common.h"
#include "utils/pitch-estimator.h"
#include <string_view>

namespace VadFilterOnnx {

bool is_ten_vad(const std::vector<const char *> &input_names,
                const std::vector<const char *> &output_names) {
    // 7 inputs:  samples, pitch, h1, c1, h2, c2, cache_features
    // 6 outputs: prob, h1, c1, h2, c2, cache_features
    //
    // Match on names rather than counts alone: a bare arity check would also
    // accept unrelated models with the same shape of I/O.
    if (input_names.size() != 7 || output_names.size() != 6) {
        return false;
    }
    return std::string_view(input_names[0]) == "samples" &&
           std::string_view(input_names[1]) == "pitch";
}

namespace {

// TEN VAD: 768-sample windows every 256 samples, two LSTMs plus a conv cache. The 41st feature
// is an F0 estimate the ONNX graph cannot compute; see utils/pitch-estimator.h.
class TenEngine final : public FrameEngine {
  public:
    TenEngine(std::shared_ptr<const OnnxModel> model, const VadConfig &config)
        : FrameEngine(256, 768),
          ort_(std::move(model)),
          pitch_estimator_(config.sample_rate, frame_shift_) {
        reset_state();
    }

  private:
    static constexpr std::array<int64_t, 2> kStateShape{ 1, 64 };
    static constexpr std::array<int64_t, 3> kCacheShape{ 1, 2, 41 };
    static constexpr std::array<int64_t, 2> kPitchShape{ 1, 1 };

    void reset_state() override {
        auto &alloc = ort_.allocator();
        ResetState(h1_, alloc, kStateShape);
        ResetState(c1_, alloc, kStateShape);
        ResetState(h2_, alloc, kStateShape);
        ResetState(c2_, alloc, kStateShape);
        ResetState(pitch_, alloc, kPitchShape);
        ResetState(conv_cache_, alloc, kCacheShape);
        pitch_estimator_.reset();
    }

    float forward(float *data) override {
        // F0 in Hz for this frame, 0.0 when unvoiced.
        pitch_.GetTensorMutableData<float>()[0] = pitch_estimator_.process(data, frame_length_);

        // Inputs: samples, pitch, h1, c1, h2, c2, cache_features
        std::vector<Ort::Value> inputs;
        inputs.reserve(7);
        inputs.push_back(ort_.wave(data, frame_length_));
        inputs.push_back(std::move(pitch_));
        inputs.push_back(std::move(h1_));
        inputs.push_back(std::move(c1_));
        inputs.push_back(std::move(h2_));
        inputs.push_back(std::move(c2_));
        inputs.push_back(std::move(conv_cache_));

        auto out = ort_.run(inputs);

        // Reclaim the pitch tensor so it can be reused next frame.
        pitch_ = std::move(inputs[1]);

        // Outputs: prob, h1, c1, h2, c2, cache_features
        h1_ = std::move(out[1]);
        c1_ = std::move(out[2]);
        h2_ = std::move(out[3]);
        c2_ = std::move(out[4]);
        conv_cache_ = std::move(out[5]);
        return out[0].GetTensorData<float>()[0];
    }

    OrtRunner ort_;
    PitchEstimator pitch_estimator_;
    Ort::Value pitch_{ nullptr };
    Ort::Value h1_{ nullptr };
    Ort::Value c1_{ nullptr };
    Ort::Value h2_{ nullptr };
    Ort::Value c2_{ nullptr };
    Ort::Value conv_cache_{ nullptr };
};

} // namespace

std::unique_ptr<VadEngine> make_ten_engine(std::shared_ptr<const OnnxModel> model,
                                           const VadConfig &config) {
    return std::make_unique<TenEngine>(std::move(model), config);
}

} // namespace VadFilterOnnx
