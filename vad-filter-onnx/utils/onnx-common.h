#pragma once
#include "onnxruntime_cxx_api.h" // NOLINT

template <typename T = float> void Fill(Ort::Value *tensor, T value) {
    auto n = tensor->GetTypeInfo().GetTensorTypeAndShapeInfo().GetElementCount();
    auto p = tensor->GetTensorMutableData<T>();
    std::fill(p, p + n, value);
}

Ort::Env &GetOrtEnv();
Ort::SessionOptions GetSessionOptions(int num_threads = 1, int device_id = -1);
std::shared_ptr<Ort::Session> ReadOnnx(const std::string &path, int num_threads = 1,
                                       int device_id = -1);
void GetInputOutputInfo(const std::shared_ptr<Ort::Session> &session,
                        std::vector<const char *> &in_names, std::vector<const char *> &out_names);