#pragma once

namespace VadFilterOnnx {

enum class VadType {
    WebrtcVad,
    SileroVadV4,
    SileroVadV5,
    FsmnVad,
    TenVad,
    None,
};

struct VadSegment {
    int idx;
    int start;
    int end;
    int start_ms;
    int end_ms;

    VadSegment(int idx = -1, int start = -1, int end = -1, int start_ms = -1, int end_ms = -1)
        : idx(idx), start(start), end(end), start_ms(start_ms), end_ms(end_ms) {}
};

struct VadConfig {
    float threshold = 0.4;
    int sample_rate = 16000;
    int speech_window_size_ms = 300;      // window size for speech detection (silence -> speech)
    int speech_window_threshold_ms = 250; // speech duration threshold within speech window
    int silence_window_size_ms = 600;     // window size for silence detection (speech -> silence)
    int silence_window_threshold_ms = 500; // silence duration threshold within silence window
    int max_speech_ms = 10000;            // max speech duration per segment
    int left_padding_ms = 100;            // padding for speech start
    int right_padding_ms = 100;           // padding for speech end
};

} // namespace VadFilterOnnx
