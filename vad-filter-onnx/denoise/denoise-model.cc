#include "denoise/denoise-model.h"
#include "denoise/gtcrn-denoise-model.h"
#include "utils/onnx-common.h"
#include <array>
#include <string_view>

namespace VadFilterOnnx {
namespace {

bool HasExpectedTensor(Ort::Session *session, std::size_t index, bool input,
                       const std::vector<int64_t> &expected_shape) {
    const auto type_info =
        input ? session->GetInputTypeInfo(index) : session->GetOutputTypeInfo(index);
    const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    return tensor_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
           tensor_info.GetShape() == expected_shape;
}

bool HasExpectedGtcrnInterface(Ort::Session *session) {
    const std::array<std::vector<int64_t>, 6> shapes = {
        std::vector<int64_t>{ 1, 256 },         std::vector<int64_t>{ 2, 1, 16, 16, 33 },
        std::vector<int64_t>{ 2, 3, 1, 1, 16 }, std::vector<int64_t>{ 2, 1, 33, 16 },
        std::vector<int64_t>{ 1, 256 },         std::vector<int64_t>{ 1, 256 },
    };
    for (std::size_t index = 0; index < shapes.size(); ++index) {
        if (!HasExpectedTensor(session, index, true, shapes[index]) ||
            !HasExpectedTensor(session, index, false, shapes[index])) {
            return false;
        }
    }
    return true;
}

bool HasGtcrnMetadata(Ort::Session *session) {
    Ort::AllocatorWithDefaultOptions allocator;
    auto value =
        session->GetModelMetadata().LookupCustomMetadataMapAllocated("model_type", allocator);
    return value && std::string_view(value.get()) == "gtcrn_denoise";
}

} // namespace

std::unique_ptr<DenoiseModel> DenoiseModel::create(const std::string &path, int num_threads,
                                                   int device_id) {
    auto session = ReadOnnx(path, num_threads, device_id);
    std::vector<const char *> input_names;
    std::vector<const char *> output_names;
    GetInputOutputInfo(session, input_names, output_names);
    if (!is_gtcrn_denoise(input_names, output_names) || !HasExpectedGtcrnInterface(session.get()) ||
        !HasGtcrnMetadata(session.get())) {
        return nullptr;
    }

    auto model = std::make_unique<GtcrnDenoiseModel>();
    model->session_ = std::move(session);
    model->input_names_ = std::move(input_names);
    model->output_names_ = std::move(output_names);
    return model;
}

} // namespace VadFilterOnnx
