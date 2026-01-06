#!/usr/bin/env bash

set -e


# 格式化 CMakeLists.txt
# find . -name "CMakeLists.txt" -exec cmake-format -i {} +

# 格式化 C++ 代码
find vad-filter-onnx -type f \( -name "*.cc" -o -name "*.h" \) -exec ./clang-format -style=file -i {} +
# exit 0


build_dir=build
cmake -S . -B ${build_dir} \
    -DCMAKE_BUILD_TYPE=release \
    -DONNXRUNTIME_FILE=./downloads/onnxruntime-linux-x64-gpu-1.17.1-patched.zip
cmake --build ${build_dir} -j16 
# cd build && make  -j16 # VERBOSE=1
