# 

# Vad models 

- silero-vad v4.0 ~ v6.2 is supported. Download models from [github releases](https://github.com/snakers4/silero-vad/releases)
- fsmn-vad and ten-vad could be found in public/ dir.

| Model | feature | Frame Length | Frame Shift  |
| :--- | :--- | :--- | :--- |
| Silero-VAD v4 | STFT | 32ms | 32ms |
| Silero-VAD v5.0 ~ v6.2 | STFT | 36ms | 32ms | 
| Fsmn-VAD | Fbank | 25ms | 10ms |
| Ten-Vad |  MelBank | 48ms | 48ms |

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
