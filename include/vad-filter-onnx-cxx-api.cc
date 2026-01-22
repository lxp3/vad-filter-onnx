#include "vad-filter-onnx-cxx-api.h"
#include "vad/vad-model.h"
#include <onnxruntime_cxx_api.h>

namespace VadFilterOnnx {

class AutoVadModel::Impl {
public:
    // Handle mode: internal_model_ is the handle from VadModel::create
    // Instance mode: internal_model_ is the instance from handle->init()
    std::unique_ptr<VadModel> internal_model_;

    Impl() = default;
    explicit Impl(std::unique_ptr<VadModel> model) : internal_model_(std::move(model)) {}
};

AutoVadModel::AutoVadModel() : impl_(std::make_unique<Impl>()) {}

AutoVadModel::~AutoVadModel() = default;

std::unique_ptr<AutoVadModel> AutoVadModel::create(const std::string &path, int num_threads, int device_id) {
    auto model = VadModel::create(path, num_threads, device_id);
    if (!model) {
        return nullptr;
    }
    // Using a private constructor via a helper since std::make_unique can't access private ctor
    struct AutoVadModelPublic : public AutoVadModel {
        AutoVadModelPublic() : AutoVadModel() {}
    };
    auto api_model = std::make_unique<AutoVadModelPublic>();
    api_model->impl_->internal_model_ = std::move(model);
    return std::move(api_model);
}

std::unique_ptr<AutoVadModel> AutoVadModel::init(const VadConfig &config) {
    if (!impl_->internal_model_) {
        return nullptr;
    }
    auto instance = impl_->internal_model_->init(config);
    if (!instance) {
        return nullptr;
    }
    struct AutoVadModelPublic : public AutoVadModel {
        AutoVadModelPublic() : AutoVadModel() {}
    };
    auto api_instance = std::make_unique<AutoVadModelPublic>();
    api_instance->impl_->internal_model_ = std::move(instance);
    return std::move(api_instance);
}

std::vector<VadSegment> AutoVadModel::decode(float *data, int n, bool input_finished) {
    if (!impl_->internal_model_) {
        return {};
    }
    return impl_->internal_model_->decode(data, n, input_finished);
}

void AutoVadModel::reset() {
    if (impl_->internal_model_) {
        impl_->internal_model_->reset();
    }
}

VadSegment AutoVadModel::flush() {
    if (impl_->internal_model_) {
        return impl_->internal_model_->flush();
    }
    return VadSegment();
}

std::vector<std::string> get_ort_available_providers() {
    return Ort::GetAvailableProviders();
}

} // namespace VadFilterOnnx

