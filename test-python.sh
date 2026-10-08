#! /bin/bash


stage=${1:-1}
build_dir=${2:-build}

export PYTHONPATH=$(pwd)/${build_dir}/python:${PYTHONPATH}

# mp3_path=public/TownTheme.mp3
# wav_path=public/TownTheme.wav
#audio_path="public/6666.09-59-02.c79b9f1c-c613-41d0-8e02-94e89ca3bca4.wav"
#wav_path="public/zh.wav/6666.09-59-02.c79b9f1c-c613-41d0-8e02-94e89ca3bca4.16k.wav"
audio_path="d66de6f4-31f3-4208-bc24-394d8e92ca90_3.wav"
if [ ${stage} -eq -1 ]; then
    ffmpeg -i ${audio_path} \
    -map_channel 0.0.0 \
    -ar 16000 \
    -ac 1 \
    -c:a pcm_s16le \
    ${wav_path}
fi

# audio_path=public/wavs/zh.wav
# silero_vad_v4_onnx_path="public/models/silero_vad.v4.onnx"
# silero_vad_v6_onnx_path="public/models/silero_vad_16k_op15.v6.onnx"
onnx_path="public/models/fsmn_vad.16k.onnx"
#onnx_path="public/models/silero_vad.v4.onnx"
if [ ${stage} -eq 1 ]; then
    python3 python/tests/test-online-vad.py \
        --model-path ${onnx_path} \
        --audio-path ${audio_path} \
        --threshold 0.8
fi

if [ ${stage} -eq 2 ]; then
    python3 python/tests/test-offline-vad.py \
        --model-path ${onnx_path} \
        --audio-path ${audio_path} \
        --threshold 0.8 
fi

# testdir=/data/user/lxp/deploy/label-studio/projects/cn/data


part="催收-信用飞-343111-20251201-20251207"
werdir=wer
# wavscp=${testdir}/${part}.scp
wavscp=/data/user/lxp/trt/asr/3-20251022-trt-infer/speech-recognizer/wavs.16k/${part}.scp
vadscp=${werdir}/${part}.vad
savedir=wer/wavs
if [ ${stage} -eq 3 ]; then
    mkdir -p ${werdir}
    rm -rf ${savedir} && mkdir -p ${savedir}
    python python/tests/test-online-vad-threads.py \
        --model-path ${onnx_path} \
        --wavscp ${wavscp} \
        --vadscp ${vadscp} \
        --save-dir ${savedir} \
        --num-threads 20
fi
