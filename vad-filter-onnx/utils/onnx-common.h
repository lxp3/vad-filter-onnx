#pragma once
#include "onnxruntime_cxx_api.h" // NOLINT
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace VadFilterOnnx {

template <typename T = float> void Fill(Ort::Value *tensor, T value) {
    auto n = tensor->GetTypeInfo().GetTensorTypeAndShapeInfo().GetElementCount();
    auto p = tensor->GetTensorMutableData<T>();
    std::fill(p, p + n, value);
}

// Gives `tensor` the given shape (creating it on first use) and zeroes it.
template <size_t N>
void ResetState(Ort::Value &tensor, Ort::AllocatorWithDefaultOptions &allocator,
                const std::array<int64_t, N> &shape) {
    if (tensor == nullptr) {
        tensor = Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
    }
    Fill<float>(&tensor, 0.0f);
}

Ort::Env &GetOrtEnv();
Ort::SessionOptions GetSessionOptions(int num_threads = 1, int device_id = -1);
std::shared_ptr<Ort::Session> ReadOnnx(const std::string &path, int num_threads = 1,
                                       int device_id = -1);
void GetInputOutputInfo(const std::shared_ptr<Ort::Session> &session,
                        std::vector<const char *> &in_names, std::vector<const char *> &out_names);

// A loaded session with its input/output names. Immutable; shared by every stream of a model.
struct OnnxModel {
    std::shared_ptr<Ort::Session> session;
    std::vector<const char *> input_names;
    std::vector<const char *> output_names;
};

// What a model engine needs to run one inference: the shared model plus per-stream allocator.
class OrtRunner {
  public:
    explicit OrtRunner(std::shared_ptr<const OnnxModel> model)
        : model_(std::move(model)),
          memory_info_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)) {}

    Ort::AllocatorWithDefaultOptions &allocator() { return allocator_; }
    size_t num_inputs() const { return model_->input_names.size(); }
    const Ort::MemoryInfo &memory_info() const { return memory_info_; }

    // [1, n] tensor viewing `data` (not copied).
    Ort::Value wave(float *data, int n) const {
        const std::array<int64_t, 2> shape = { 1, n };
        return Ort::Value::CreateTensor(memory_info_, data, n, shape.data(), shape.size());
    }

    // 1-element int64 tensor viewing `*value`.
    Ort::Value scalar(int64_t *value) const {
        int64_t shape = 1;
        return Ort::Value::CreateTensor(memory_info_, value, 1, &shape, 1);
    }

    std::vector<Ort::Value> run(std::vector<Ort::Value> &inputs) const {
        return model_->session->Run(Ort::RunOptions{ nullptr }, model_->input_names.data(),
                                    inputs.data(), inputs.size(), model_->output_names.data(),
                                    model_->output_names.size());
    }

  private:
    std::shared_ptr<const OnnxModel> model_;
    Ort::MemoryInfo memory_info_;
    Ort::AllocatorWithDefaultOptions allocator_;
};

} // namespace VadFilterOnnx