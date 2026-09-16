#! /bin/bash

vad_build_dir=${PWD}/build_shared
bin=${vad_build_dir}/test-vad-threads
model_dir=${PWD}/public/models

seconds=${seconds:-60}
webrtc_concurrencys=(100 500 1000 3000 5000)
onnx_concurrencys=(100 500 1000 1500 2000 2500 3000)
vad_models=()
shopt -s nullglob
vad_models+=("${model_dir}"/*vad*.onnx)

run_test() {
    local model_path=$1
    local concurrency=$2
    local extra_args=()
    if [[ ${model_path} == *'.8k.'* ]]; then
        extra_args+=(--vad-sample-rate 8000)
    fi
    echo "===== model=${model_path} concurrency=${concurrency} seconds=${seconds} ====="
    ${bin} \
        --model-path ${model_path} \
        --concurrency ${concurrency} \
        --seconds ${seconds} \
        "${extra_args[@]}"
}

for n in "${webrtc_concurrencys[@]}"; do
    run_test webrtc ${n}
done

for model in "${vad_models[@]}"; do
    for n in "${onnx_concurrencys[@]}"; do
        run_test ${model} ${n}
    done
done
