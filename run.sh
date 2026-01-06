#! /bin/bash

stage=$1

# mp3_path=public/TownTheme.mp3
# wav_path=public/TownTheme.wav
audio_path="public/6666.09-59-02.c79b9f1c-c613-41d0-8e02-94e89ca3bca4.wav"
wav_path="public/6666.09-59-02.c79b9f1c-c613-41d0-8e02-94e89ca3bca4.16k.wav"
if [ ${stage} -eq -1 ]; then
    ffmpeg -i ${audio_path} \
    -map_channel 0.0.0 \
    -ar 16000 \
    -ac 1 \
    -c:a pcm_s16le \
    ${wav_path}
fi

silero_vad_v4_onnx_path="public/silero_vad.v4.onnx"
fsmn_vad_onnx_path="public/fsmn_vad.16k.onnx"
if [ ${stage} -eq 0 ]; then
    echo "Testing FSMN VAD model..."
    ./build/test-vad-online-decode \
        --model-path ${fsmn_vad_onnx_path} \
        --wav-path ${wav_path} \
        --sample-rate 16000 \
        --threshold 0.6 \
        --chunk-size-ms 100 \
        --max-speech-ms 5000
fi