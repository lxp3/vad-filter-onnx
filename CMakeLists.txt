cmake_minimum_required(VERSION 3.15)
project(vad_project LANGUAGES CXX)

find_package(Threads REQUIRED)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.21")
    set(VAD_FILTER_ONNX_IS_TOP_LEVEL ${PROJECT_IS_TOP_LEVEL})
else()
    if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
        set(VAD_FILTER_ONNX_IS_TOP_LEVEL ON)
    else()
        set(VAD_FILTER_ONNX_IS_TOP_LEVEL OFF)
    endif()
endif()

option(BUILD_SHARED_LIBS "Whether to build shared libraries" ON)
option(ENABLE_GPU "Enable GPU support for ONNX Runtime" OFF)
option(ENABLE_PYTHON "Enable Python bindings" OFF)
option(VAD_FILTER_ONNX_BUILD_EXAMPLES "Build vad-filter-onnx example programs" ${VAD_FILTER_ONNX_IS_TOP_LEVEL})
option(VAD_FILTER_ONNX_BUILD_TESTS "Build vad-filter-onnx test programs" ${VAD_FILTER_ONNX_IS_TOP_LEVEL})
set(GLIBCXX_USE_CXX11_ABI "0" CACHE STRING "Value for _GLIBCXX_USE_CXX11_ABI")

if(NOT BUILD_SHARED_LIBS STREQUAL "ON" AND ENABLE_GPU STREQUAL "ON")
    message(FATAL_ERROR "ENABLE_GPU=ON is not supported when BUILD_SHARED_LIBS is not ON. Please set ENABLE_GPU=OFF or build with BUILD_SHARED_LIBS=ON.")
endif()

# Enable Position Independent Code (PIC) for static libraries
# This is required on Linux when linking static libs into shared objects (like Python modules)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 修复 RuntimeLibrary 不匹配问题 (MT vs MD)
if(MSVC)
    if(BUILD_SHARED_LIBS STREQUAL "ON")
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    else()
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
endif()


if(MSVC)
    add_compile_options(/utf-8)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加cmake模块路径
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Dependencies
include(onnxruntime)

# Add subdirectories
add_subdirectory(vad-filter-onnx)

if(ENABLE_PYTHON)
    add_subdirectory(python/src python)
endif()

if(VAD_FILTER_ONNX_BUILD_EXAMPLES)
    # Binaries
    add_executable(
        test-vad-online-decode
        vad-filter-onnx/bin/test-vad-online-decode.cc
        vad-filter-onnx/bin/resample.cc
    )
    target_link_libraries(test-vad-online-decode PRIVATE vad_filter_onnx)
    target_include_directories(test-vad-online-decode PRIVATE
        vad-filter-onnx/include
        vad-filter-onnx/vad
    )

    add_executable(test-rtf-online vad-filter-onnx/bin/test-rtf-online.cc)
    target_link_libraries(test-rtf-online PRIVATE vad_filter_onnx)
    target_include_directories(test-rtf-online PRIVATE vad-filter-onnx/include)

    add_executable(test-rtf-offline vad-filter-onnx/bin/test-rtf-offline.cc)
    target_link_libraries(test-rtf-offline PRIVATE vad_filter_onnx)
    target_include_directories(test-rtf-offline PRIVATE vad-filter-onnx/include)

    add_executable(test-rtf-denoise-online vad-filter-onnx/bin/test-rtf-denoise-online.cc)
    target_link_libraries(test-rtf-denoise-online PRIVATE vad_filter_onnx)
    target_include_directories(test-rtf-denoise-online PRIVATE vad-filter-onnx/include)

    add_executable(test-rtf-denoise-offline vad-filter-onnx/bin/test-rtf-denoise-offline.cc)
    target_link_libraries(test-rtf-denoise-offline PRIVATE vad_filter_onnx)
    target_include_directories(test-rtf-denoise-offline PRIVATE vad-filter-onnx/include)

    add_executable(test-denoise vad-filter-onnx/bin/test-denoise.cc vad-filter-onnx/bin/resample.cc)
    target_link_libraries(test-denoise PRIVATE vad_filter_onnx)
    target_include_directories(test-denoise PRIVATE vad-filter-onnx/include vad-filter-onnx/utils)

    # Installation
    install(TARGETS test-vad-online-decode test-rtf-online test-rtf-offline
        test-rtf-denoise-online test-rtf-denoise-offline test-denoise
        RUNTIME DESTINATION bin
    )
endif()

# Install ONNX Runtime shared library
if(BUILD_SHARED_LIBS STREQUAL "ON")
    if(WIN32)
        file(GLOB ORT_LIBS "${ONNXRUNTIME_ROOT_DIR}/lib/*.dll")
        install(FILES ${ORT_LIBS} DESTINATION bin)
    elseif(APPLE)
        file(GLOB ORT_LIBS "${ONNXRUNTIME_ROOT_DIR}/lib/*.dylib")
        install(FILES ${ORT_LIBS} DESTINATION lib)
    else()
        file(GLOB ORT_LIBS "${ONNXRUNTIME_ROOT_DIR}/lib/*.so*")
        install(FILES ${ORT_LIBS} DESTINATION lib)
    endif()
else()
    install(FILES "${ONNXRUNTIME_LIB}" DESTINATION lib)
endif()
