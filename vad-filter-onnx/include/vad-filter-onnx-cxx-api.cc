#include "vad-filter-onnx-cxx-api.h"
#include "vad/vad-model.h"
#include <onnxruntime_cxx_api.h>
#include <stdexcept>

namespace VadFilterOnnx {

// An AutoVadModel is either a handle (owns `model`) or an instance (owns `stream`).
class AutoVadModel::Impl {
  public:
    std::unique_ptr<VadModel> model;
    std::unique_ptr<VadStream> stream;

    VadStream &require_stream(const char *what) const {
        if (!stream) {
            throw std::runtime_error(std::string(what) +
                                     " is only supported on an initialized model instance "
                                     "returned by init().");
        }
        return *stream;
    }
};

AutoVadModel::AutoVadModel() : impl_(std::make_unique<Impl>()) {}

AutoVadModel::~AutoVadModel() = default;

std::unique_ptr<AutoVadModel> AutoVadModel::create(const std::string &path, int num_threads,
                                                   int device_id) {
    auto model = VadModel::create(path, num_threads, device_id);
    if (!model) {
        return nullptr;
    }
    // The constructor is private; a local class inside a member function may call it.
    struct Creatable : AutoVadModel {};
    auto api = std::make_unique<Creatable>();
    api->impl_->model = std::move(model);
    return api;
}

std::unique_ptr<AutoVadModel> AutoVadModel::create_webrtc() {
    auto model = VadModel::create_webrtc();
    if (!model) {
        return nullptr;
    }
    struct Creatable : AutoVadModel {};
    auto api = std::make_unique<Creatable>();
    api->impl_->model = std::move(model);
    return api;
}

std::unique_ptr<AutoVadModel> AutoVadModel::init(const VadConfig &config) {
    if (!impl_->model) {
        return nullptr;
    }
    auto stream = impl_->model->init(config);
    if (!stream) {
        return nullptr;
    }
    struct Creatable : AutoVadModel {};
    auto api = std::make_unique<Creatable>();
    api->impl_->stream = std::move(stream);
    return api;
}

std::vector<VadSegment> AutoVadModel::decode(float *data, int n, bool input_finished) {
    if (!impl_->stream) {
        return {};
    }
    return impl_->stream->decode(data, n, input_finished);
}

void AutoVadModel::setup_config(const VadConfig &config) {
    impl_->require_stream("setup_config").setup_config(config);
}

VadConfig AutoVadModel::get_config() const { return impl_->require_stream("get_config").config(); }

void AutoVadModel::reset() {
    if (impl_->stream) {
        impl_->stream->reset();
    }
}

VadSegment AutoVadModel::flush() {
    if (impl_->stream) {
        return impl_->stream->flush();
    }
    return VadSegment();
}

std::vector<std::string> get_ort_available_providers() { return Ort::GetAvailableProviders(); }

} // namespace VadFilterOnnx
