#pragma once

#include <memory>
#include <string>
#include <vector>
#include "vad-config.h"

namespace VadFilterOnnx {

/**
 * @brief A high-level C++ API for the VAD model using the Pimpl idiom.
 * The signatures match VadFilterOnnx::VadModel for consistency.
 */
class AutoVadModel {
public:
    /**
     * @brief Create a model handle (loads ONNX session).
     * @param path Path to the ONNX model.
     * @param num_threads Number of threads for ONNX Runtime.
     * @param device_id Device ID (-1 for CPU, >=0 for GPU).
     * @return Unique pointer to AutoVadModel handle.
     */
    static std::unique_ptr<AutoVadModel> create(const std::string &path, int num_threads = 1, int device_id = -1);

    /**
     * @brief Initialize a model instance for inference.
     * @param config VAD configuration.
     * @return Unique pointer to AutoVadModel instance.
     */
    std::unique_ptr<AutoVadModel> init(const VadConfig &config);

    /**
     * @brief Process audio data.
     * @param data Pointer to PCM data.
     * @param n Number of samples.
     * @param input_finished End of stream flag.
     * @return Detected segments.
     */
    std::vector<VadSegment> decode(float *data, int n, bool input_finished);

    void reset();
    VadSegment flush();

    ~AutoVadModel();

private:
    AutoVadModel(); // Private constructor used by factory
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace VadFilterOnnx
