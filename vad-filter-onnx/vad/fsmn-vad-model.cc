#include "vad/fsmn-vad-model.h"
#include <algorithm>
#include <stdexcept>
#include <string_view>

namespace VadFilterOnnx {

bool is_fsmn_vad(const std::vector<const char *> &input_names,
                 const std::vector<const char *> &output_names) {
    if (input_names.size() == 6 && output_names.size() == 5 &&
        std::string_view(input_names[0]) == "speech" &&
        std::string_view(input_names[1]) == "in_cache0" &&
        std::string_view(input_names[2]) == "in_cache1" &&
        std::string_view(input_names[3]) == "in_cache2" &&
        std::string_view(input_names[4]) == "in_cache3" &&
        std::string_view(input_names[5]) == "padding" &&
        std::string_view(output_names[0]) == "logits") {
        return true;
    }
    return false;
}

namespace {

constexpr int kMaxNewFrames = 10;

class FsmnEngine final : public VadEngine {
  public:
    FsmnEngine(std::shared_ptr<const OnnxModel> model, int sample_rate)
        : ort_(std::move(model)),
          frame_shift_(10 * (sample_rate / 1000)),
          frame_length_(25 * (sample_rate / 1000)) {
        reset();
    }

    int frame_shift() const override { return frame_shift_; }

    void reset() override {
        is_first_inference_ = true;
        reminder_.clear();
        reminder_offset_ = 0;
        for (auto &cache : caches_)
            ResetState(cache, ort_.allocator(), kCacheShape);
    }

    void process(float *data, int n, bool input_finished, std::vector<float> &probs) override {
        // Bound each ONNX invocation to roughly 100 ms. Four FBank frames
        // (55 ms) remain at the buffer front between invocations for LFR context.
        const size_t max_inference_samples =
            static_cast<size_t>(kMaxNewFrames - 1) * frame_shift_ + frame_length_;
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
            const int min_frames = is_first_inference_ ? 3 : 5;
            const size_t min_samples =
                static_cast<size_t>(min_frames - 1) * frame_shift_ + frame_length_;

            if (final_block) {
                const size_t final_min_samples =
                    is_first_inference_ ? static_cast<size_t>(frame_length_)
                                        : static_cast<size_t>(3 * frame_shift_ + frame_length_);
                if (available >= final_min_samples) {
                    int32_t first_padding = is_first_inference_ ? 2 : 0;
                    forward_frames(reminder_.data() + reminder_offset_, static_cast<int>(available),
                                   first_padding, 2, probs);
                    is_first_inference_ = false;
                }
                reminder_.clear();
                reminder_offset_ = 0;
                break;
            }

            if (available < min_samples ||
                (available < max_inference_samples && all_input_buffered)) {
                break;
            }

            // First block maps to [2, 0], middle blocks to [0, 0]. The first
            // block emits two left-padded logits, so it consumes two fewer shifts.
            int32_t first_padding = is_first_inference_ ? 2 : 0;
            const size_t produced = forward_frames(reminder_.data() + reminder_offset_,
                                                   static_cast<int>(available), first_padding, 0,
                                                   probs);
            size_t consumed_frames = produced - (is_first_inference_ ? 2 : 0);
            reminder_offset_ += consumed_frames * frame_shift_;
            is_first_inference_ = false;
        }

        if (input_finished) {
            reminder_.clear();
            reminder_offset_ = 0;
        }
    }

  private:
    static constexpr std::array<int64_t, 4> kCacheShape{ 1, 128, 19, 1 };

    // Runs the model; appends the speech probability of every output frame, returns their count.
    size_t forward_frames(float *data, int n, int32_t first_p, int32_t last_p,
                          std::vector<float> &probs) {
        std::array<int32_t, 2> padding_values = { first_p, last_p };
        std::array<int64_t, 1> padding_shape = { 2 };

        std::vector<Ort::Value> inputs;
        inputs.reserve(6);
        inputs.push_back(ort_.wave(data, n));
        for (auto &cache : caches_)
            inputs.push_back(std::move(cache));
        inputs.push_back(Ort::Value::CreateTensor<int32_t>(ort_.memory_info(), padding_values.data(),
                                                           padding_values.size(),
                                                           padding_shape.data(),
                                                           padding_shape.size()));

        auto out = ort_.run(inputs);

        // Update internal caches for the next streaming chunk
        for (size_t i = 0; i < caches_.size(); ++i)
            caches_[i] = std::move(out[i + 1]);

        // logits is the noise probability: FunASR gives -1 < 1 - 2 * p_noise < 1, here
        // 0 < 2 * p_noise - 1 < 1, so p_speech = 1 - p_noise.
        const float *logits = out[0].GetTensorData<float>();
        const size_t frames = static_cast<size_t>(out[0].GetTensorTypeAndShapeInfo().GetShape()[1]);
        for (size_t i = 0; i < frames; ++i)
            probs.push_back(1 - logits[i]);
        return frames;
    }

    OrtRunner ort_;
    int frame_shift_;
    int frame_length_;
    std::array<Ort::Value, 4> caches_{ Ort::Value{ nullptr }, Ort::Value{ nullptr },
                                       Ort::Value{ nullptr }, Ort::Value{ nullptr } };
    bool is_first_inference_ = true;
    std::vector<float> reminder_;
    size_t reminder_offset_ = 0;
};

} // namespace

std::unique_ptr<VadEngine> make_fsmn_engine(std::shared_ptr<const OnnxModel> model,
                                            const VadConfig &config) {
    if (config.sample_rate < 1000) {
        throw std::runtime_error("sample_rate must be a positive multiple of 1000");
    }
    return std::make_unique<FsmnEngine>(std::move(model), config.sample_rate);
}

} // namespace VadFilterOnnx
