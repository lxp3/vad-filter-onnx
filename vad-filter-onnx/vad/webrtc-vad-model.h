#pragma once

#include "vad/vad-engine.h"
#include "vad-config.h"
#include <memory>

namespace VadFilterOnnx {

void ValidateWebrtcVadConfig(const VadConfig &config);
int WebrtcVadFrameSamples(const VadConfig &config);

// WebRTC VAD (libfvad) needs no model file. Throws if the config is not valid for WebRTC.
std::unique_ptr<VadEngine> make_webrtc_engine(const VadConfig &config);

} // namespace VadFilterOnnx
