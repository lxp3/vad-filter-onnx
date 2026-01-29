cmake_minimum_required(VERSION 3.15)
project(vad_project LANGUAGES CXX)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
option(BUILD_SHARED_LIBS "Whether to build shared libraries" ON)
option(ENABLE_GPU "Enable GPU support for ONNX Runtime" OFF)
option(ENABLE_PYTHON "Enable Python bindings" OFF)

if(NOT BUILD_SHARED_LIBS STREQUAL "ON" AND ENABLE_GPU STREQUAL "ON")
    message(FATAL_ERROR "ENABLE_GPU=ON is not supported when BUILD_SHARED_LIBS is not ON. Please set ENABLE_GPU=OFF or build with BUILD_SHARED_LIBS=ON.")
endif()

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

# Binaries
add_executable(test-vad-online-decode vad-filter-onnx/bin/test-vad-online-decode.cc)
target_link_libraries(test-vad-online-decode PRIVATE vad_filter_onnx_lib)
target_include_directories(test-vad-online-decode PRIVATE 
    vad-filter-onnx/include
    vad-filter-onnx/vad
)

# Installation
install(TARGETS test-vad-online-decode RUNTIME DESTINATION bin)

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
endif()

