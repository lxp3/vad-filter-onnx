#include "denoise-filter-onnx-cxx-api.h"
#include "denoise/denoise-model.h"

namespace VadFilterOnnx {

class AutoDenoiseModel::Impl {
  public:
    std::unique_ptr<DenoiseModel> internal_model_;
};

AutoDenoiseModel::AutoDenoiseModel() : impl_(std::make_unique<Impl>()) {}
AutoDenoiseModel::~AutoDenoiseModel() = default;

std::unique_ptr<AutoDenoiseModel> AutoDenoiseModel::create(const std::string &path, int num_threads,
                                                           int device_id) {
    auto model = DenoiseModel::create(path, num_threads, device_id);
    if (!model) {
        return nullptr;
    }
    struct PublicModel : public AutoDenoiseModel {};
    auto result = std::make_unique<PublicModel>();
    result->impl_->internal_model_ = std::move(model);
    return result;
}

std::unique_ptr<AutoDenoiseModel> AutoDenoiseModel::init(const DenoiseConfig &config) {
    if (!impl_->internal_model_) {
        return nullptr;
    }
    auto model = impl_->internal_model_->init(config);
    if (!model) {
        return nullptr;
    }
    struct PublicModel : public AutoDenoiseModel {};
    auto result = std::make_unique<PublicModel>();
    result->impl_->internal_model_ = std::move(model);
    return result;
}

std::vector<float> AutoDenoiseModel::decode(const float *data, int n, bool input_finished) {
    if (!impl_->internal_model_) {
        return {};
    }
    return impl_->internal_model_->decode(data, n, input_finished);
}

void AutoDenoiseModel::reset() {
    if (impl_->internal_model_) {
        impl_->internal_model_->reset();
    }
}

} // namespace VadFilterOnnx
