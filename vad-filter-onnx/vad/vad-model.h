#pragma once

#include "vad-config.h"
#include "vad/segmenter.h"
#include "vad/vad-engine.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace VadFilterOnnx {

// Builds a fresh engine for a stream. Captures the shared, read-only model resources.
using EngineFactory = std::function<std::unique_ptr<VadEngine>(const VadConfig &)>;

// One independent audio stream. Created by VadModel::init(); not thread safe.
class VadStream {
  public:
    VadStream(EngineFactory factory, const VadConfig &config);

    std::vector<VadSegment> decode(float *data, int n, bool input_finished);
    VadSegment flush() { return segmenter_.flush(); }
    void reset();
    // Applies a new config and resets the stream. sample_rate must not change.
    void setup_config(const VadConfig &config);
    const VadConfig &config() const { return segmenter_.config(); }

  private:
    EngineFactory factory_;
    std::unique_ptr<VadEngine> engine_;
    Segmenter segmenter_;
    std::vector<float> probs_; // scratch: probabilities produced by one decode() call
};

// Loaded model resources, shared by all streams created from it.
class VadModel {
  public:
    // Loads an ONNX model and detects its type. Returns nullptr for unknown models.
    static std::unique_ptr<VadModel> create(const std::string &path, int num_threads = 1,
                                            int device_id = -1);
    static std::unique_ptr<VadModel> create_webrtc();

    std::unique_ptr<VadStream> init(const VadConfig &config) const;

  private:
    explicit VadModel(EngineFactory factory) : factory_(std::move(factory)) {}

    EngineFactory factory_;
};

} // namespace VadFilterOnnx
