#include "vad/nemo-marblenet-vad-model.h"
#include <algorithm>
#include <stdexcept>
#include <string_view>

namespace VadFilterOnnx {

bool is_nemo_marblenet_vad(const std::vector<const char *> &input_names,
                           const std::vector<const char *> &output_names) {
    return input_names.size() == 1 && output_names.size() == 1 &&
           std::string_view(input_names[0]) == "speech" &&
           std::string_view(output_names[0]) == "probs";
}

namespace {

// Output frame granularity: encoder's stride-2 first block over 10ms mel
// hops yields one probability every 20ms == 320 samples @16kHz.
constexpr int kOutputFrameSamples = 320;
// Receptive field measured via gradient/impulse-response probing of the
// exported encoder: a middle output frame's support spans roughly +-145
// mel frames (+-1.45s); 1.5s of context on each side is a safe margin.
constexpr size_t kContextSamples = 24000; // 1.5 s
// New samples confirmed per inference call: 300 ms == 15 output frames.
constexpr size_t kChunkShiftSamples = 4800;
// Minimum samples the exported ONNX graph can produce a frame for.
constexpr size_t kMinWindowSamples = 400;

class NemoMarbleNetEngine final : public VadEngine {
  public:
    explicit NemoMarbleNetEngine(std::shared_ptr<const OnnxModel> model)
        : ort_(std::move(model)) {}

    int frame_shift() const override { return kOutputFrameSamples; }

    void reset() override {
        buffer_.clear();
        dropped_total_ = 0;
        confirmed_samples_ = 0;
    }

    void process(float *data, int n, bool input_finished, std::vector<float> &probs) override {
        // buffer_[0] always corresponds to absolute sample index
        // (confirmed_samples_ - kContextSamples), i.e. it retains exactly the left context
        // needed for the next window; trim anything older every iteration below.
        buffer_.insert(buffer_.end(), data, data + n);

        while (true) {
            // Samples in buffer_ beyond confirmed_samples_ that have not yet been
            // confirmed as output frames.
            size_t confirmed_offset = confirmed_samples_ - dropped_total_;
            size_t new_available = buffer_.size() - confirmed_offset;

            bool final_block = input_finished; // buffer_ already holds everything appended so far

            if (!final_block && new_available < kChunkShiftSamples) {
                break;
            }
            if (new_available == 0) {
                break;
            }

            size_t shift_samples = final_block ? new_available : kChunkShiftSamples;

            size_t left_context = std::min(confirmed_offset, kContextSamples);
            size_t window_start = confirmed_offset - left_context;
            size_t window_end =
                std::min(buffer_.size(), confirmed_offset + shift_samples + kContextSamples);
            size_t window_len = window_end - window_start;

            if (window_len < kMinWindowSamples) {
                if (!final_block) {
                    break;
                }
                // Too little data left to produce even one frame; drop the tail.
                buffer_.clear();
                break;
            }

            auto window_probs =
                forward_window(buffer_.data() + window_start, static_cast<int>(window_len));

            size_t keep_start_frame = left_context / static_cast<size_t>(kOutputFrameSamples);
            size_t available_frames =
                window_probs.size() > keep_start_frame ? window_probs.size() - keep_start_frame : 0;
            size_t keep_count = final_block
                                    ? available_frames
                                    : shift_samples / static_cast<size_t>(kOutputFrameSamples);
            keep_count = std::min(keep_count, available_frames);

            probs.insert(probs.end(), window_probs.begin() + keep_start_frame,
                         window_probs.begin() + keep_start_frame + keep_count);
            confirmed_samples_ += keep_count * kOutputFrameSamples;

            // Drop everything before the new left-context start for the next window.
            size_t new_confirmed_offset = confirmed_samples_ - dropped_total_;
            size_t drop_count = new_confirmed_offset > kContextSamples
                                    ? new_confirmed_offset - kContextSamples
                                    : 0;
            if (drop_count > 0) {
                buffer_.erase(buffer_.begin(), buffer_.begin() + drop_count);
                dropped_total_ += drop_count;
            }

            if (final_block) {
                buffer_.clear();
                break;
            }
        }

        if (input_finished) {
            buffer_.clear();
        }
    }

  private:
    std::vector<float> forward_window(float *data, int n) {
        std::vector<Ort::Value> inputs;
        inputs.push_back(ort_.wave(data, n));
        auto out = ort_.run(inputs);

        const float *probs_ptr = out[0].GetTensorData<float>();
        const size_t frames = static_cast<size_t>(out[0].GetTensorTypeAndShapeInfo().GetShape()[1]);
        return std::vector<float>(probs_ptr, probs_ptr + frames);
    }

    OrtRunner ort_;
    std::vector<float> buffer_;
    // Total samples ever erased from the front of buffer_; buffer_[i]
    // corresponds to absolute sample index (dropped_total_ + i).
    size_t dropped_total_ = 0;
    // Samples whose output frames have been emitted so far.
    size_t confirmed_samples_ = 0;
};

} // namespace

std::unique_ptr<VadEngine> make_nemo_marblenet_engine(std::shared_ptr<const OnnxModel> model,
                                                      const VadConfig &config) {
    if (config.sample_rate != 16000) {
        throw std::runtime_error("NeMo MarbleNet VAD supports 16000 Hz audio only");
    }
    return std::make_unique<NemoMarbleNetEngine>(std::move(model));
}

} // namespace VadFilterOnnx
