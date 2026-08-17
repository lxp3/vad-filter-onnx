#include "denoise/deepfilternet-denoise-model.h"
#include "denoise/denoise-model.h"
#include "denoise/dfsmn-ans-psm-48k-denoise-model.h"
#include "denoise/dpdfnet-denoise-model.h"
#include "denoise/frcrn-se-16k-denoise-model.h"
#include "denoise/gtcrn-denoise-model.h"
#include "denoise/mossformer2-se-48k-denoise-model.h"
#include "denoise/mossformergan-se-16k-denoise-model.h"
#include "denoise/resemble-enhance-denoiser-denoise-model.h"
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

bool HasSingleWaveformIoInterface(Ort::Session *session) {
    const auto speech_shape = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    const auto enhanced_shape =
        session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    // The batch dim is declared symbolic (not a static 1) on the enhanced
    // output because onnxsim's shape inference cannot prove it stays 1
    // through the model's internal Squeeze/Unsqueeze ops, even though it
    // always is at runtime (this backend never uses batch>1). Accept -1
    // (dynamic) as well as a static 1 on both input and output.
    if (speech_shape.size() != 2 || (speech_shape[0] != 1 && speech_shape[0] != -1)) {
        return false;
    }
    if (enhanced_shape.size() != 2 || (enhanced_shape[0] != 1 && enhanced_shape[0] != -1)) {
        return false;
    }
    return session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType() ==
               ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
           session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType() ==
               ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

bool HasFrcrnSe16kMetadata(Ort::Session *session, int *sample_rate) {
    Ort::AllocatorWithDefaultOptions allocator;
    auto model_type =
        session->GetModelMetadata().LookupCustomMetadataMapAllocated("model_type", allocator);
    if (!model_type || std::string_view(model_type.get()) != "frcrn_se_16k_denoise") {
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

bool HasMossformerganSe16kMetadata(Ort::Session *session, int *sample_rate) {
    Ort::AllocatorWithDefaultOptions allocator;
    auto model_type =
        session->GetModelMetadata().LookupCustomMetadataMapAllocated("model_type", allocator);
    if (!model_type || std::string_view(model_type.get()) != "mossformergan_se_16k_denoise") {
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

bool HasMossformer2Se48kMetadata(Ort::Session *session, int *sample_rate) {
    Ort::AllocatorWithDefaultOptions allocator;
    auto model_type =
        session->GetModelMetadata().LookupCustomMetadataMapAllocated("model_type", allocator);
    if (!model_type || std::string_view(model_type.get()) != "mossformer2_se_48k_denoise") {
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

bool HasDeepfilternetMetadata(Ort::Session *session, int *sample_rate, std::size_t *state_size,
                              std::size_t *hop_size, std::size_t *delay_hops) {
    Ort::AllocatorWithDefaultOptions allocator;
    auto model_type =
        session->GetModelMetadata().LookupCustomMetadataMapAllocated("model_type", allocator);
    if (!model_type) {
        return false;
    }
    const std::string_view type(model_type.get());
    if (type != "deepfilternet_denoise" && type != "deepfilternet2_denoise" &&
        type != "deepfilternet3_denoise") {
        return false;
    }
    auto rate_str =
        session->GetModelMetadata().LookupCustomMetadataMapAllocated("sample_rate", allocator);
    auto state_size_str =
        session->GetModelMetadata().LookupCustomMetadataMapAllocated("state_size", allocator);
    auto frame_shift_str =
        session->GetModelMetadata().LookupCustomMetadataMapAllocated("frame_shift", allocator);
    auto delay_hops_str =
        session->GetModelMetadata().LookupCustomMetadataMapAllocated("delay_hops", allocator);
    if (!rate_str || !state_size_str || !frame_shift_str) {
        return false;
    }
    *sample_rate = std::atoi(rate_str.get());
    *state_size = static_cast<std::size_t>(std::atoll(state_size_str.get()));
    *hop_size = static_cast<std::size_t>(std::atoll(frame_shift_str.get()));
    *delay_hops = delay_hops_str ? static_cast<std::size_t>(std::atoll(delay_hops_str.get())) : 1;
    return *sample_rate > 0 && *state_size > 0 && *hop_size > 0;
}

bool HasDfsmnAnsPsm48kMetadata(Ort::Session *session, int *sample_rate, std::size_t *state_size,
                               std::size_t *hop_size, std::size_t *delay_hops) {
    Ort::AllocatorWithDefaultOptions allocator;
    auto model_type =
        session->GetModelMetadata().LookupCustomMetadataMapAllocated("model_type", allocator);
    if (!model_type || std::string_view(model_type.get()) != "dfsmn_ans_psm_48k_denoise") {
        return false;
    }
    auto rate_str =
        session->GetModelMetadata().LookupCustomMetadataMapAllocated("sample_rate", allocator);
    auto state_size_str =
        session->GetModelMetadata().LookupCustomMetadataMapAllocated("state_size", allocator);
    auto frame_shift_str =
        session->GetModelMetadata().LookupCustomMetadataMapAllocated("frame_shift", allocator);
    auto delay_hops_str =
        session->GetModelMetadata().LookupCustomMetadataMapAllocated("delay_hops", allocator);
    if (!rate_str || !state_size_str || !frame_shift_str) {
        return false;
    }
    *sample_rate = std::atoi(rate_str.get());
    *state_size = static_cast<std::size_t>(std::atoll(state_size_str.get()));
    *hop_size = static_cast<std::size_t>(std::atoll(frame_shift_str.get()));
    *delay_hops = delay_hops_str ? static_cast<std::size_t>(std::atoll(delay_hops_str.get())) : 1;
    return *sample_rate > 0 && *state_size > 0 && *hop_size > 0;
}

bool HasResembleEnhanceDenoiserMetadata(Ort::Session *session, int *sample_rate) {
    Ort::AllocatorWithDefaultOptions allocator;
    auto model_type =
        session->GetModelMetadata().LookupCustomMetadataMapAllocated("model_type", allocator);
    if (!model_type || std::string_view(model_type.get()) != "resemble_enhance_denoiser_denoise") {
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

    int frcrn_sample_rate = 0;
    if (is_frcrn_se_16k_denoise(input_names, output_names) && HasSingleWaveformIoInterface(session.get()) &&
        HasFrcrnSe16kMetadata(session.get(), &frcrn_sample_rate)) {
        auto model = std::make_unique<FrcrnSe16kDenoiseModel>();
        model->set_sample_rate(frcrn_sample_rate);
        model->session_ = std::move(session);
        model->input_names_ = std::move(input_names);
        model->output_names_ = std::move(output_names);
        return model;
    }

    int mossformergan_sample_rate = 0;
    if (is_mossformergan_se_16k_denoise(input_names, output_names) &&
        HasSingleWaveformIoInterface(session.get()) &&
        HasMossformerganSe16kMetadata(session.get(), &mossformergan_sample_rate)) {
        auto model = std::make_unique<MossformerganSe16kDenoiseModel>();
        model->set_sample_rate(mossformergan_sample_rate);
        model->session_ = std::move(session);
        model->input_names_ = std::move(input_names);
        model->output_names_ = std::move(output_names);
        return model;
    }

    int mossformer2_sample_rate = 0;
    if (is_mossformer2_se_48k_denoise(input_names, output_names) &&
        HasSingleWaveformIoInterface(session.get()) &&
        HasMossformer2Se48kMetadata(session.get(), &mossformer2_sample_rate)) {
        auto model = std::make_unique<Mossformer2Se48kDenoiseModel>();
        model->set_sample_rate(mossformer2_sample_rate);
        model->session_ = std::move(session);
        model->input_names_ = std::move(input_names);
        model->output_names_ = std::move(output_names);
        return model;
    }

    int resemble_enhance_sample_rate = 0;
    if (is_resemble_enhance_denoiser_denoise(input_names, output_names) &&
        HasSingleWaveformIoInterface(session.get()) &&
        HasResembleEnhanceDenoiserMetadata(session.get(), &resemble_enhance_sample_rate)) {
        auto model = std::make_unique<ResembleEnhanceDenoiserDenoiseModel>();
        model->set_sample_rate(resemble_enhance_sample_rate);
        model->session_ = std::move(session);
        model->input_names_ = std::move(input_names);
        model->output_names_ = std::move(output_names);
        return model;
    }

    int dfsmn_ans_psm_48k_sample_rate = 0;
    std::size_t dfsmn_ans_psm_48k_state_size = 0;
    std::size_t dfsmn_ans_psm_48k_hop_size = 0;
    std::size_t dfsmn_ans_psm_48k_delay_hops = 1;
    if (is_dfsmn_ans_psm_48k_denoise(input_names, output_names) &&
        HasExpectedDpdfnetInterface(session.get(), &dfsmn_ans_psm_48k_state_size,
                                    &dfsmn_ans_psm_48k_hop_size) &&
        HasDfsmnAnsPsm48kMetadata(session.get(), &dfsmn_ans_psm_48k_sample_rate,
                                  &dfsmn_ans_psm_48k_state_size, &dfsmn_ans_psm_48k_hop_size,
                                  &dfsmn_ans_psm_48k_delay_hops)) {
        auto model = std::make_unique<DfsmnAnsPsm48kDenoiseModel>();
        model->set_state_size(dfsmn_ans_psm_48k_state_size);
        model->set_hop_size(dfsmn_ans_psm_48k_hop_size);
        model->set_sample_rate(dfsmn_ans_psm_48k_sample_rate);
        model->set_delay_hops(dfsmn_ans_psm_48k_delay_hops);
        model->session_ = std::move(session);
        model->input_names_ = std::move(input_names);
        model->output_names_ = std::move(output_names);
        return model;
    }

    int deepfilternet_sample_rate = 0;
    std::size_t deepfilternet_state_size = 0;
    std::size_t deepfilternet_hop_size = 0;
    std::size_t deepfilternet_delay_hops = 1;
    if (is_deepfilternet_denoise(input_names, output_names) &&
        HasExpectedDpdfnetInterface(session.get(), &deepfilternet_state_size,
                                    &deepfilternet_hop_size) &&
        HasDeepfilternetMetadata(session.get(), &deepfilternet_sample_rate,
                                 &deepfilternet_state_size, &deepfilternet_hop_size,
                                 &deepfilternet_delay_hops)) {
        auto model = std::make_unique<DeepFilterNetDenoiseModel>();
        model->set_state_size(deepfilternet_state_size);
        model->set_hop_size(deepfilternet_hop_size);
        model->set_sample_rate(deepfilternet_sample_rate);
        model->set_delay_hops(deepfilternet_delay_hops);
        model->session_ = std::move(session);
        model->input_names_ = std::move(input_names);
        model->output_names_ = std::move(output_names);
        return model;
    }

    return nullptr;
}

} // namespace VadFilterOnnx
