include(FetchContent)

set(VAD_FILTER_ONNX_GIT_REPOSITORY
    "https://github.com/your-org/vad-filter-onnx.git"
    CACHE STRING
    "Git repository used to fetch vad-filter-onnx"
)
set(VAD_FILTER_ONNX_GIT_TAG
    "main"
    CACHE STRING
    "Git tag, branch, or commit used to fetch vad-filter-onnx"
)

set(ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
set(VAD_FILTER_ONNX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(VAD_FILTER_ONNX_BUILD_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    vad_filter_onnx
    GIT_REPOSITORY "${VAD_FILTER_ONNX_GIT_REPOSITORY}"
    GIT_TAG        "${VAD_FILTER_ONNX_GIT_TAG}"
)

FetchContent_MakeAvailable(vad_filter_onnx)
