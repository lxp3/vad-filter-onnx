#pragma once

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
    int window_size_ms = 300;
    int min_speech_ms = 250;
    int max_speech_ms = 10000;
    int left_padding_ms = 100;
    int right_padding_ms = 100;
<<<<<<< HEAD
};
=======
};
>>>>>>> dbc5f0b83d32c046afc13174a5ce1e77e23094ed
