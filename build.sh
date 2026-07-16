#!/usr/bin/env bash

set -e

export http_proxy="http://192.168.58.72:7890"
export https_proxy="http://192.168.58.72:7890" 
# Configuration
BUILD_SHARED_LIBS="ON"
GLIBCXX_USE_CXX11_ABI="${1:-1}"

if [ "$BUILD_SHARED_LIBS" = "ON" ]; then
    BUILD_DIR="build_shared"
else
    BUILD_DIR="build_static"
fi

# Print configuration info
echo -e "\033[0;36mConfiguring project ($BUILD_DIR)...\033[0m"
echo -e "\033[0;36m_GLIBCXX_USE_CXX11_ABI=$GLIBCXX_USE_CXX11_ABI\033[0m"

# Build options
# -DCMAKE_BUILD_TYPE=Release is required for single-config generators (like Unix Makefiles)
# ENABLE_GPU=ON to use GPU version of ONNX Runtime as specified in onnxruntime.cmake
cmake -B "$BUILD_DIR" -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS="$BUILD_SHARED_LIBS" \
    -DGLIBCXX_USE_CXX11_ABI="$GLIBCXX_USE_CXX11_ABI" \
    -DENABLE_GPU=OFF \
    -DENABLE_PYTHON=ON \
    -DVAD_FILTER_ONNX_BUILD_EXAMPLES=OFF \
    -DVAD_FILTER_ONNX_BUILD_TESTS=OFF

echo -e "\n\033[0;36m--- Building vad-filter-onnx ---\033[0m"

# Build the project using all available cores
cmake --build "$BUILD_DIR" --config Release -j$(nproc)

echo -e "\033[0;32mBuild completed successfully!\033[0m"
