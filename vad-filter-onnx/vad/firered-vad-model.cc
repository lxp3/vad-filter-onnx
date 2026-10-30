#include "vad/firered-vad-model.h"
#include <algorithm>
#include <stdexcept>
#include <string_view>

namespace VadFilterOnnx {

bool is_firered_vad(const std::vector<const char *> &input_names,
                    const std::vector<const char *> &output_names) {
    return input_names.size() == 2 && output_names.size() == 2 &&
           std::string_view(input_names[0]) == "speech" &&
           std::string_view(input_names[1]) == "caches_in" &&
           std::string_view(output_names[0]) == "probs" &&
           std::string_view(output_names[1]) == "caches_out";
}

namespace {

constexpr int kFrameShift = 160;  // 10 ms at 16 kHz
constexpr int kFrameLength = 400; // 25 ms at 16 kHz
constexpr int kMaxFramesPerInference = 10;

class FireredEngine final : public VadEngine {
  public:
    explicit FireredEngine(std::shared_ptr<const OnnxModel> model) : ort_(std::move(model)) {
        reset();
    }

    int frame_shift() const override { return kFrameShift; }

    void reset() override {
        reminder_.clear();
        reminder_offset_ = 0;
        ResetState(caches_, ort_.allocator(), kCacheShape);
    }

    void process(float *data, int n, bool input_finished, std::vector<float> &probs) override {
        constexpr size_t max_inference_samples =
            static_cast<size_t>(kMaxFramesPerInference - 1) * kFrameShift + kFrameLength;
        size_t input_offset = 0;

        while (input_offset < static_cast<size_t>(n) ||
               (input_finished && reminder_.size() > reminder_offset_)) {
            if (reminder_offset_ > 0 && (reminder_offset_ >= max_inference_samples ||
                                         reminder_offset_ * 2 >= reminder_.size())) {
                reminder_.erase(reminder_.begin(), reminder_.begin() + reminder_offset_);
                reminder_offset_ = 0;
            }

            size_t available = reminder_.size() - reminder_offset_;
            size_t room = max_inference_samples - std::min(available, max_inference_samples);
            size_t append_count = std::min(room, static_cast<size_t>(n) - input_offset);
            if (append_count > 0) {
                reminder_.insert(reminder_.end(), data + input_offset,
                                 data + input_offset + append_count);
            }
            input_offset += append_count;
            available += append_count;

            const bool all_input_buffered = input_offset == static_cast<size_t>(n);
            const bool final_block = input_finished && all_input_buffered;
            if (available < static_cast<size_t>(kFrameLength)) {
                break;
            }
            if (!final_block && available < max_inference_samples && all_input_buffered) {
                break;
            }

            int num_frames = (static_cast<int>(available) - kFrameLength) / kFrameShift + 1;
            int num_samples = kFrameLength + (num_frames - 1) * kFrameShift;
            const size_t produced =
                forward_frames(reminder_.data() + reminder_offset_, num_samples, probs);

            if (final_block) {
                reminder_.clear();
                reminder_offset_ = 0;
                break;
            }
            reminder_offset_ += produced * kFrameShift;
        }

        if (input_finished) {
            reminder_.clear();
            reminder_offset_ = 0;
        }
    }

  private:
    static constexpr std::array<int64_t, 4> kCacheShape{ 8, 1, 128, 19 };

    // Runs the model over n samples; appends one probability per frame, returns their count.
    size_t forward_frames(float *data, int n, std::vector<float> &probs) {
        std::vector<Ort::Value> inputs;
        inputs.reserve(2);
        inputs.push_back(ort_.wave(data, n));
        inputs.push_back(std::move(caches_));

        auto out = ort_.run(inputs);

        caches_ = std::move(out[1]);

        const float *out_probs = out[0].GetTensorData<float>();
        const size_t frames = static_cast<size_t>(out[0].GetTensorTypeAndShapeInfo().GetShape()[1]);
        probs.insert(probs.end(), out_probs, out_probs + frames);
        return frames;
    }

    OrtRunner ort_;
    Ort::Value caches_{ nullptr };
    std::vector<float> reminder_;
    size_t reminder_offset_ = 0;
};

} // namespace

std::unique_ptr<VadEngine> make_firered_engine(std::shared_ptr<const OnnxModel> model,
                                               const VadConfig &config) {
    if (config.sample_rate != 16000) {
        throw std::runtime_error("FireRedVAD supports 16000 Hz audio only");
    }
    return std::make_unique<FireredEngine>(std::move(model));
}

} // namespace VadFilterOnnx
