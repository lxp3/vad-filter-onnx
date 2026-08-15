#include "denoise/denoise-model.h"
#include "denoise/dpdfnet-denoise-model.h"
#include "denoise/gtcrn-denoise-model.h"
#include "utils/onnx-common.h"
#include <array>
#include <cstdlib>
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

bool HasExpectedDpdfnetInterface(Ort::Session *session, std::size_t *state_size,
                                 std::size_t *hop_size) {
    const auto speech_shape = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (speech_shape.size() != 2 || speech_shape[0] != 1 || speech_shape[1] <= 0) {
        return false;
    }
    for (std::size_t index = 0; index < 3; ++index) {
        if (!HasExpectedTensor(session, index, true, speech_shape) ||
            !HasExpectedTensor(session, index, false, speech_shape)) {
            return false;
        }
    }
    const auto state_in_shape =
        session->GetInputTypeInfo(3).GetTensorTypeAndShapeInfo().GetShape();
    const auto state_out_shape =
        session->GetOutputTypeInfo(3).GetTensorTypeAndShapeInfo().GetShape();
    if (state_in_shape.size() != 1 || state_in_shape != state_out_shape ||
        state_in_shape[0] <= 0) {
        return false;
    }
    if (session->GetInputTypeInfo(3).GetTensorTypeAndShapeInfo().GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        session->GetOutputTypeInfo(3).GetTensorTypeAndShapeInfo().GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        return false;
    }
    *state_size = static_cast<std::size_t>(state_in_shape[0]);
    *hop_size = static_cast<std::size_t>(speech_shape[1]);
    return true;
}

bool HasDpdfnetMetadata(Ort::Session *session, int *sample_rate) {
    Ort::AllocatorWithDefaultOptions allocator;
    auto model_type =
        session->GetModelMetadata().LookupCustomMetadataMapAllocated("model_type", allocator);
    if (!model_type || std::string_view(model_type.get()) != "dpdfnet_denoise") {
        return false;
    }
    auto rate_str =
        session->GetModelMetadata().LookupCustomMetadataMapAllocated("sample_rate", allocator);
    if (!rate_str) {
        return false;
    }
    *sample_rate = std::atoi(rate_str.get());
    return *sample_rate > 0;
}

} // namespace

std::unique_ptr<DenoiseModel> DenoiseModel::create(const std::string &path, int num_threads,
                                                   int device_id) {
    auto session = ReadOnnx(path, num_threads, device_id);
    std::vector<const char *> input_names;
    std::vector<const char *> output_names;
    GetInputOutputInfo(session, input_names, output_names);

    if (is_gtcrn_denoise(input_names, output_names) && HasExpectedGtcrnInterface(session.get()) &&
        HasGtcrnMetadata(session.get())) {
        auto model = std::make_unique<GtcrnDenoiseModel>();
        model->session_ = std::move(session);
        model->input_names_ = std::move(input_names);
        model->output_names_ = std::move(output_names);
        return model;
    }

    std::size_t state_size = 0;
    std::size_t hop_size = 0;
    int sample_rate = 0;
    if (is_dpdfnet_denoise(input_names, output_names) &&
        HasExpectedDpdfnetInterface(session.get(), &state_size, &hop_size) &&
        HasDpdfnetMetadata(session.get(), &sample_rate)) {
        auto model = std::make_unique<DpdfnetDenoiseModel>();
        model->set_state_size(state_size);
        model->set_hop_size(hop_size);
        model->set_sample_rate(sample_rate);
        model->session_ = std::move(session);
        model->input_names_ = std::move(input_names);
        model->output_names_ = std::move(output_names);
        return model;
    }

    return nullptr;
}

} // namespace VadFilterOnnx
