project(vad_lib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files
file(GLOB_RECURSE SOURCES 
    "denoise/*.cc"
    "utils/*.cc"
    "vad/*.cc"
    "include/*.cc"
)
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/vad/test-sliding-window-bit.cc")
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/denoise/test-gtcrn-denoise.cc")
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/denoise/test-dpdfnet-denoise.cc")
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/denoise/test-frcrn-se-16k-denoise.cc")
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/denoise/test-mossformergan-se-16k-denoise.cc")
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/denoise/test-mossformer2-se-48k-denoise.cc")
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/denoise/test-resemble-enhance-denoiser-denoise.cc")
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/denoise/test-deepfilternet-denoise.cc")
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/denoise/test-dfsmn-ans-psm-48k-denoise.cc")


# Library
if(BUILD_SHARED_LIBS STREQUAL "ON")
    add_library(vad_filter_onnx SHARED ${SOURCES})
else()
    add_library(vad_filter_onnx STATIC ${SOURCES})
endif()
add_library(vad_filter_onnx::vad_filter_onnx ALIAS vad_filter_onnx)

target_compile_features(vad_filter_onnx PUBLIC cxx_std_20)
target_compile_definitions(vad_filter_onnx PUBLIC _GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI})

target_include_directories(vad_filter_onnx PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(BUILD_SHARED_LIBS STREQUAL "ON")
    target_link_libraries(vad_filter_onnx PRIVATE onnxruntime)
else()
    target_link_libraries(vad_filter_onnx PUBLIC onnxruntime)
endif()

if(VAD_FILTER_ONNX_BUILD_TESTS)
    # Test
    add_executable(test-sliding-window-bit vad/test-sliding-window-bit.cc)
    target_link_libraries(test-sliding-window-bit PRIVATE vad_filter_onnx)

    add_executable(test-gtcrn-denoise denoise/test-gtcrn-denoise.cc)
    target_link_libraries(test-gtcrn-denoise PRIVATE vad_filter_onnx)

    add_executable(test-dpdfnet-denoise denoise/test-dpdfnet-denoise.cc)
    target_link_libraries(test-dpdfnet-denoise PRIVATE vad_filter_onnx)

    add_executable(test-frcrn-se-16k-denoise denoise/test-frcrn-se-16k-denoise.cc)
    target_link_libraries(test-frcrn-se-16k-denoise PRIVATE vad_filter_onnx)

    add_executable(test-mossformergan-se-16k-denoise denoise/test-mossformergan-se-16k-denoise.cc)
    target_link_libraries(test-mossformergan-se-16k-denoise PRIVATE vad_filter_onnx)

    add_executable(test-mossformer2-se-48k-denoise denoise/test-mossformer2-se-48k-denoise.cc)
    target_link_libraries(test-mossformer2-se-48k-denoise PRIVATE vad_filter_onnx)

    add_executable(test-resemble-enhance-denoiser-denoise denoise/test-resemble-enhance-denoiser-denoise.cc)
    target_link_libraries(test-resemble-enhance-denoiser-denoise PRIVATE vad_filter_onnx)

    add_executable(test-deepfilternet-denoise denoise/test-deepfilternet-denoise.cc)
    target_link_libraries(test-deepfilternet-denoise PRIVATE vad_filter_onnx)

    add_executable(test-dfsmn-ans-psm-48k-denoise denoise/test-dfsmn-ans-psm-48k-denoise.cc)
    target_link_libraries(test-dfsmn-ans-psm-48k-denoise PRIVATE vad_filter_onnx)
endif()

# Installation
install(TARGETS vad_filter_onnx
    EXPORT ${PROJECT_NAME}Targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)
