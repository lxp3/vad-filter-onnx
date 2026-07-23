# 

# Vad models

The FireRedVAD exports were compared with the source PyTorch Stream-VAD model
using one second of deterministic 16 kHz float audio (`torch.manual_seed(20260723)`)
and zero-initialized caches. The FSMN-VAD exports used the corresponding 8 kHz
or 16 kHz source PyTorch model, the same duration and seed, zero-initialized
caches, and feature padding `[2, 2]`. The table reports the maximum absolute
error over the probability/logits output and all output caches.

RTF benchmarks use deterministic audio, 5 warmup runs, and 20 measured runs on
an Intel Xeon Silver 4316 CPU. Online RTF uses 5 seconds of audio and 100 ms
chunks; offline RTF is reported for both 5-second and 120-second inputs.

```bash
./build/test-rtf-online \
  --model-path public/models/fsmn_vad.16k.onnx \
  --chunk-ms 100 \
  --num-warmups 5 \
  --num-runs 20
```

| Model | Sample rate | Feature | Frame Length | Frame Shift | Logits max diff | Cache max diff | Online RTF (5s) | Offline RTF (5s) | Offline RTF (120s) | Model address |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| FireRed-VAD float | 16000 | Fbank | 25ms | 10ms | 0.00000417 | 0.00030851 | 0.011287 | 0.011907 | 0.011891 | [`firered_vad.onnx`](public/models/firered_vad.onnx) |
| FireRed-VAD int8 | 16000 | Fbank | 25ms | 10ms | 0.05357799 | 5.96644974 | 0.010993 | 0.011226 | 0.011194 | [`firered_vad.int8.onnx`](public/models/firered_vad.int8.onnx) |
| FSMN-VAD 16k float | 16000 | Fbank | 25ms | 10ms | 0.00000522 | 0.00002837 | 0.005762 | 0.008597 | 0.008684 | [`fsmn_vad.16k.onnx`](public/models/fsmn_vad.16k.onnx) |
| FSMN-VAD 16k int8 | 16000 | Fbank | 25ms | 10ms | 0.07808840 | 0.35685480 | 0.005494 | 0.008536 | 0.008140 | [`fsmn_vad.16k.int8.onnx`](public/models/fsmn_vad.16k.int8.onnx) |
| FSMN-VAD 8k float | 8000 | Fbank | 25ms | 10ms | 0.00000000 | 0.00000127 | 0.004503 | 0.006307 | 0.006321 | [`fsmn_vad.8k.onnx`](public/models/fsmn_vad.8k.onnx) |
| FSMN-VAD 8k int8 | 8000 | Fbank | 25ms | 10ms | 0.00000150 | 0.01216167 | 0.003612 | 0.005645 | 0.005662 | [`fsmn_vad.8k.int8.onnx`](public/models/fsmn_vad.8k.int8.onnx) |
| Silero-VAD v4 | 16000 | STFT | 32ms | 32ms | - | - | 0.005879 | 0.005400 | 0.005450 | [`silero_vad.v4.onnx`](public/models/silero_vad.v4.onnx) |
| Silero-VAD v5 | 16000 | STFT | 36ms | 32ms | - | - | 0.004727 | 0.004806 | 0.004792 | [`silero_vad.v5.onnx`](public/models/silero_vad.v5.onnx) |
| Silero-VAD v6 | 16000 | STFT | 36ms | 32ms | - | - | 0.004745 | 0.004851 | 0.004693 | [`silero_vad.v6.onnx`](public/models/silero_vad.v6.onnx) |
| Silero-VAD v6 opset 15 | 16000 | STFT | 36ms | 32ms | - | - | 0.004717 | 0.004659 | 0.004600 | [`silero_vad_16k_op15.v6.onnx`](public/models/silero_vad_16k_op15.v6.onnx) |
| Ten-VAD | 16000 | MelBank | 48ms | 48ms | - | - | 0.006124 | 0.006074 | 0.006118 | [`ten_vad.onnx`](public/models/ten_vad.onnx) |

## C++ CMake integration

The recommended way to use this project from another C++ CMake project is
`FetchContent`. Link the exported target `vad_filter_onnx::vad_filter_onnx`.

```cmake
include(FetchContent)

set(ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
set(VAD_FILTER_ONNX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(VAD_FILTER_ONNX_BUILD_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    vad_filter_onnx
    GIT_REPOSITORY https://github.com/your-org/vad-filter-onnx.git
    GIT_TAG        main
)

FetchContent_MakeAvailable(vad_filter_onnx)

target_link_libraries(your_app PRIVATE vad_filter_onnx::vad_filter_onnx)
```

Minimal C++ usage:

```cpp
#include <vad-filter-onnx-cxx-api.h>

int main() {
    auto handle = VadFilterOnnx::AutoVadModel::create("public/models/fsmn_vad.16k.onnx");
    VadFilterOnnx::VadConfig config;
    auto vad = handle->init(config);
    return vad ? 0 : 1;
}
```

ONNX Runtime is downloaded automatically by default. The archive cache is kept
under the build directory at `_deps/onnxruntime-downloads`. Override it with
`-DVAD_FILTER_ONNX_ORT_DOWNLOAD_DIR=/path/to/cache` if needed.

If you prefer a reusable wrapper, copy `cmake/vad_filter_onnx.cmake` into your
project, set `VAD_FILTER_ONNX_GIT_REPOSITORY` and `VAD_FILTER_ONNX_GIT_TAG`, and
include it before linking `vad_filter_onnx::vad_filter_onnx`.
