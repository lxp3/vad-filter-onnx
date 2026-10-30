#include "vad/vad-model.h"
#include "utils/onnx-common.h"
#include "vad/firered-vad-model.h"
#include "vad/fsmn-vad-model.h"
#include "vad/nemo-marblenet-vad-model.h"
#include "vad/pulse-vad-model.h"
#include "vad/silero-vad-model.h"
#include "vad/ten-vad-model.h"
#include "vad/webrtc-vad-model.h"
#include <cstdio>
#include <stdexcept>

namespace VadFilterOnnx {

namespace {

using EngineMaker = std::unique_ptr<VadEngine> (*)(std::shared_ptr<const OnnxModel>,
                                                   const VadConfig &);

// Picks the engine maker for a loaded ONNX model; nullptr for unknown models.
EngineMaker detect(const OnnxModel &model, const char *&name) {
    const auto &in = model.input_names;
    const auto &out = model.output_names;
    if (is_pulse_vad(model.session.get())) {
        name = "PulseVad";
        return make_pulse_engine;
    }
    if (is_silero_vad_v4(in, out)) {
        name = "SileroVadV4";
        return make_silero_v4_engine;
    }
    if (is_silero_vad_v5(in, out)) {
        name = "SileroVadV5";
        return make_silero_v5_engine;
    }
    if (is_fsmn_vad(in, out)) {
        name = "FsmnVad";
        return make_fsmn_engine;
    }
    if (is_ten_vad(in, out)) {
        name = "TenVad";
        return make_ten_engine;
    }
    if (is_firered_vad(in, out)) {
        name = "FireRedVad";
        return make_firered_engine;
    }
    if (is_nemo_marblenet_vad(in, out)) {
        name = "NemoMarbleNetVad";
        return make_nemo_marblenet_engine;
    }
    return nullptr;
}

} // namespace

std::unique_ptr<VadModel> VadModel::create(const std::string &path, int num_threads,
                                           int device_id) {
    auto model = std::make_shared<OnnxModel>();
    model->session = ReadOnnx(path, num_threads, device_id);
    GetInputOutputInfo(model->session, model->input_names, model->output_names);

    const char *name = nullptr;
    const EngineMaker make = detect(*model, name);
    if (make == nullptr) {
        printf("ERROR: Unknown Vad model type in %s\n", path.c_str());
        return nullptr;
    }
    printf("Success to create %s model from %s\n", name, path.c_str());

    std::shared_ptr<const OnnxModel> shared = std::move(model);
    return std::unique_ptr<VadModel>(
        new VadModel([shared, make](const VadConfig &config) { return make(shared, config); }));
}

std::unique_ptr<VadModel> VadModel::create_webrtc() {
    printf("Success to create WebrtcVad model\n");
    return std::unique_ptr<VadModel>(new VadModel(make_webrtc_engine));
}

std::unique_ptr<VadStream> VadModel::init(const VadConfig &config) const {
    return std::make_unique<VadStream>(factory_, config);
}

namespace {
Segmenter make_segmenter(const VadEngine &engine, const VadConfig &config) {
    return Segmenter(config, engine.frame_shift(), engine.threshold(config));
}
} // namespace

VadStream::VadStream(EngineFactory factory, const VadConfig &config)
    : factory_(std::move(factory)),
      engine_(factory_(config)),
      segmenter_(make_segmenter(*engine_, config)) {}

void VadStream::reset() {
    engine_->reset();
    segmenter_.reset();
}

void VadStream::setup_config(const VadConfig &config) {
    if (config.sample_rate != segmenter_.config().sample_rate) {
        throw std::runtime_error("Changing sample_rate via setup_config is not supported; "
                                 "please create a new model instance with init().");
    }
    // The frame size (WebRTC) and thresholds depend on the config, so build both anew.
    auto engine = factory_(config);
    Segmenter segmenter = make_segmenter(*engine, config);
    engine_ = std::move(engine);
    segmenter_ = std::move(segmenter);
}

std::vector<VadSegment> VadStream::decode(float *data, int n, bool input_finished) {
    if (n == 0 && !input_finished) {
        return {};
    }
    segmenter_.add_samples(n);
    probs_.clear();
    engine_->process(data, n, input_finished, probs_);
    segmenter_.push(probs_.data(), probs_.size());
    if (input_finished) {
        segmenter_.flush();
    }
    return segmenter_.take();
}

} // namespace VadFilterOnnx
