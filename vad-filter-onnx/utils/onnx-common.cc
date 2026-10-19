#include "utils/onnx-common.h"
#include <sstream>

namespace VadFilterOnnx {

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
        printf("[onnx] Initialize session in cuda:%d\n", device_id);
    } else {
        sess_opts.SetIntraOpNumThreads(num_threads); // 同一算子内部平行
        sess_opts.SetInterOpNumThreads(num_threads); // 不同操作之间并行
        sess_opts.DisableCpuMemArena();              //
        printf("[onnx] Initialize session in cpu\n");
    }

    return std::move(sess_opts);
}

std::shared_ptr<Ort::Session> ReadOnnx(const std::string &path, int num_threads, int device_id) {
    printf("[onnx] Reading onnx model: %s\n", path.c_str());
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
        printf("[onnx] Success to load onnx model: %s\n", path.c_str());
    } catch (std::exception const &e) {
        printf("[onnx] Error when load onnx model: %s\n", e.what());
        exit(0);
    }
    return std::move(session);
}

const char *DataTypeToString(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "Float";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return "Uint8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return "Int8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return "Uint16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return "Int16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "Int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "Int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: return "String";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return "Bool";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "Float16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return "Double";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return "Uint32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: return "Uint64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return "BFloat16";
        default: return "Unknown";
    }
}

void GetInputOutputInfo(const std::shared_ptr<Ort::Session> &session,
                        std::vector<const char *> &in_names, std::vector<const char *> &out_names) {
    static Ort::AllocatorWithDefaultOptions allocator;
    static std::vector<Ort::AllocatedStringPtr> allocated_names{};

    auto print_info = [&](const char *label, int i, bool is_input) {
        Ort::AllocatedStringPtr name_shared_ptr =
            is_input ? session->GetInputNameAllocated(i, allocator)
                     : session->GetOutputNameAllocated(i, allocator);
        allocated_names.push_back(std::move(name_shared_ptr));
        char *name = allocated_names.back().get();

        Ort::TypeInfo type_info =
            is_input ? session->GetInputTypeInfo(i) : session->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        std::vector<int64_t> node_dims = tensor_info.GetShape();

        std::stringstream shape;
        shape << "[";
        for (size_t j = 0; j < node_dims.size(); ++j) {
            shape << node_dims[j];
            if (j < node_dims.size() - 1) {
                shape << ", ";
            }
        }
        shape << "]";

        printf("[onnx] %s %d: name=%s, %s, dims=%s\n", label, i, name, DataTypeToString(type),
               shape.str().c_str());
        return name;
    };

    // Input info
    int num_inputs = static_cast<int>(session->GetInputCount());
    in_names.resize(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
        in_names[i] = print_info("Input", i, true);
    }

    // Output info
    int num_outputs = static_cast<int>(session->GetOutputCount());
    out_names.resize(num_outputs);
    for (int i = 0; i < num_outputs; ++i) {
        out_names[i] = print_info("Output", i, false);
    }
}

std::string LookupCustomModelMetaData(const Ort::ModelMetadata &meta_data, const char *key,
                                      OrtAllocator *allocator) {
    auto v = meta_data.LookupCustomMetadataMapAllocated(key, allocator);
    return v ? v.get() : "";
}
} // namespace VadFilterOnnx
