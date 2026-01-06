#include "utils/onnx-common.h"
#include <format>
#include <sstream>

Ort::Env &GetOrtEnv() {
    static Ort::Env env{ nullptr };
    if (!env) {
        env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "global_env");
    }
    return env;
}

// Note(lxp): device_id指定运行设备
Ort::SessionOptions GetSessionOptions(int num_threads, int device_id) {
    static std::vector<std::string> available_providers = Ort::GetAvailableProviders();
    static bool is_cuda_available = false;
    for (const auto &provider : available_providers) {
        if (provider == "CUDAExecutionProvider") {
            is_cuda_available = true;
            break;
        }
    }

    Ort::SessionOptions sess_opts;
    sess_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (device_id > 0 && is_cuda_available) {
        OrtCUDAProviderOptions config;
        config.device_id = device_id;
        config.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
        sess_opts.AppendExecutionProvider_CUDA(config);
        printf("INFO: Initialize session in cuda:%d\n", device_id);
    } else {
        sess_opts.SetIntraOpNumThreads(num_threads); // 同一算子内部平行
        sess_opts.SetInterOpNumThreads(num_threads); // 不同操作之间并行
        sess_opts.DisableCpuMemArena();              //
        printf("INFO: Initialize session in cpu\n");
    }

    return std::move(sess_opts);
}

std::shared_ptr<Ort::Session> ReadOnnx(const std::string &path, int num_threads, int device_id) {
    printf("INFO: Reading onnx model: %s\n", path.c_str());
    auto &env = GetOrtEnv();
    auto sess_opts = GetSessionOptions(num_threads, device_id);
    std::shared_ptr<Ort::Session> session{ nullptr };
    try {
#ifdef _WIN32
        // Windows需要宽字符路径
        std::wstring wide_path(path.begin(), path.end());
        session = std::make_shared<Ort::Session>(env, wide_path.c_str(), sess_opts);
#else
        session = std::make_shared<Ort::Session>(env, path.c_str(), sess_opts);
#endif
        printf("INFO: Success to load onnx model: %s\n", path.c_str());
    } catch (std::exception const &e) {
        printf("ERROR: Error when load onnx model: %s\n", e.what());
        exit(0);
    }
    return std::move(session);
}

void GetInputOutputInfo(const std::shared_ptr<Ort::Session> &session,
                        std::vector<const char *> &in_names, std::vector<const char *> &out_names) {
    static Ort::AllocatorWithDefaultOptions allocator;
    static std::vector<Ort::AllocatedStringPtr> allocated_names{};
    // Input info
    int num_nodes = session->GetInputCount();
    in_names.resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        Ort::AllocatedStringPtr name_shared_ptr = session->GetInputNameAllocated(i, allocator);
        allocated_names.push_back(std::move(name_shared_ptr));
        char *name = allocated_names.back().get();
        Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        std::vector<int64_t> node_dims = tensor_info.GetShape();
        std::stringstream shape;
        for (auto j : node_dims) {
            shape << j;
            shape << " ";
        }
        printf("INFO: \tInput %d : name=%s type=%d dims=%s\n", i, name, type, shape.str().c_str());
        in_names[i] = name;
    }
    // Output info
    num_nodes = session->GetOutputCount();
    out_names.resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        Ort::AllocatedStringPtr name_shared_ptr = session->GetOutputNameAllocated(i, allocator);
        allocated_names.push_back(std::move(name_shared_ptr));
        char *name = allocated_names.back().get();
        Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        std::vector<int64_t> node_dims = tensor_info.GetShape();
        std::stringstream shape;
        for (auto j : node_dims) {
            shape << j;
            shape << " ";
        }
        printf("INFO: \tOutput %d : name=%s type=%d dims=%s\n", i, name, type, shape.str().c_str());
        out_names[i] = name;
    }
}

std::string LookupCustomModelMetaData(const Ort::ModelMetadata &meta_data, const char *key,
                                      OrtAllocator *allocator) {
// Note(fangjun): We only tested 1.17.1 and 1.11.0
// For other versions, we may need to change it
#if ORT_API_VERSION >= 12
    auto v = meta_data.LookupCustomMetadataMapAllocated(key, allocator);
    return v ? v.get() : "";
#else
    auto v = meta_data.LookupCustomMetadataMap(key, allocator);
    std::string ans = v ? v : "";
    allocator->Free(allocator, v);
    return ans;
#endif
}