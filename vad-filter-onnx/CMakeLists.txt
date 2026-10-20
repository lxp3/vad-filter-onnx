project(vad_lib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files
file(GLOB_RECURSE SOURCES 
    "denoise/*.cc"
    "utils/*.cc"
    "vad/*.cc"
    "include/*.cc"
    "separate/*.cc"
)
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/vad/test-sliding-window-bit.cc")
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/vad/test-webrtc-vad.cc")
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/denoise/test-gtcrn-denoise.cc")
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/denoise/test-dpdfnet-denoise.cc")
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/denoise/test-frcrn-se-16k-denoise.cc")
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/denoise/test-mossformergan-se-16k-denoise.cc")
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/denoise/test-mossformer2-se-48k-denoise.cc")
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/denoise/test-resemble-enhance-denoiser-denoise.cc")
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/denoise/test-deepfilternet-denoise.cc")
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/denoise/test-dfsmn-ans-psm-48k-denoise.cc")
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/denoise/test-hush-denoise.cc")
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/separate/test-bs-roformer-separate.cc")


# Library
if(VAD_FILTER_ONNX_BUILD_SHARED_LIBS STREQUAL "ON")
    add_library(vad_filter_onnx SHARED ${SOURCES})
else()
    add_library(vad_filter_onnx STATIC ${SOURCES})
endif()
add_library(vad_filter_onnx::vad_filter_onnx ALIAS vad_filter_onnx)

target_compile_features(vad_filter_onnx PUBLIC cxx_std_20)
target_compile_definitions(vad_filter_onnx PUBLIC _GLIBCXX_USE_CXX11_ABI=${VAD_FILTER_ONNX_GLIBCXX_USE_CXX11_ABI})

target_include_directories(vad_filter_onnx PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(VAD_FILTER_ONNX_BUILD_SHARED_LIBS STREQUAL "ON")
    target_link_libraries(vad_filter_onnx PRIVATE onnxruntime fvad)
else()
    target_link_libraries(vad_filter_onnx PUBLIC onnxruntime fvad)
endif()

if(VAD_FILTER_ONNX_BUILD_TESTS)
    # Test
    add_executable(test-sliding-window-bit vad/test-sliding-window-bit.cc)
    target_link_libraries(test-sliding-window-bit PRIVATE vad_filter_onnx)

    add_executable(test-webrtc-vad vad/test-webrtc-vad.cc)
    target_link_libraries(test-webrtc-vad PRIVATE vad_filter_onnx fvad onnxruntime)

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

    add_executable(test-hush-denoise denoise/test-hush-denoise.cc)
    target_link_libraries(test-hush-denoise PRIVATE vad_filter_onnx)

    add_executable(test-bs-roformer-separate separate/test-bs-roformer-separate.cc)
    target_link_libraries(test-bs-roformer-separate PRIVATE vad_filter_onnx)
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

# Collect build artifacts into <repo>/release/{lib,include} after each build
set(VAD_FILTER_ONNX_RELEASE_DIR "${CMAKE_SOURCE_DIR}/release" CACHE PATH "Output dir for lib/ and include/")
add_custom_command(TARGET vad_filter_onnx POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${VAD_FILTER_ONNX_RELEASE_DIR}/lib" "${VAD_FILTER_ONNX_RELEASE_DIR}/include"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:vad_filter_onnx>" "${VAD_FILTER_ONNX_RELEASE_DIR}/lib/"
    COMMAND ${CMAKE_COMMAND} -DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}/include -DDST_DIR=${VAD_FILTER_ONNX_RELEASE_DIR}/include
            -P "${CMAKE_SOURCE_DIR}/cmake/copy_headers.cmake"
    VERBATIM
)
