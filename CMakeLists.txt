cmake_minimum_required(VERSION 3.10)
project(vad_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加cmake模块路径
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Dependencies
include(onnxruntime)

# Add subdirectories
add_subdirectory(vad-filter-onnx)

# Binaries
add_executable(test-vad-online-decode bin/test-vad-online-decode.cc)
target_link_libraries(test-vad-online-decode PRIVATE vad_filter_onnx)
target_include_directories(test-vad-online-decode PRIVATE include)

