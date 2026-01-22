cmake_minimum_required(VERSION 3.15)
project(vad_project LANGUAGES CXX)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# 修复 RuntimeLibrary 不匹配问题 (MT vs MD)
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
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
target_link_libraries(test-vad-online-decode PRIVATE vad_filter_onnx)
target_include_directories(test-vad-online-decode PRIVATE vad-filter-onnx/include)

# Installation
install(TARGETS test-vad-online-decode RUNTIME DESTINATION bin)

# Install ONNX Runtime shared library
if(WIN32)
    file(GLOB ORT_DLLS "${ONNXRUNTIME_ROOT}/lib/*.dll")
    install(FILES ${ORT_DLLS} DESTINATION bin)
elseif(APPLE)
    file(GLOB ORT_DLLS "${ONNXRUNTIME_ROOT}/lib/*.dylib")
    install(FILES ${ORT_DLLS} DESTINATION lib)
else()
    file(GLOB ORT_DLLS "${ONNXRUNTIME_ROOT}/lib/*.so*")
    install(FILES ${ORT_DLLS} DESTINATION lib)
endif()

