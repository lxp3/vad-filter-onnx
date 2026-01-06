#! /bin/bash

stage=$1


local_dir="models"

project=iic/speech_fsmn_vad_zh-cn-8k-common
if [ ${stage} -eq 0 ]; then
    modelscope download --model $project --local_dir ${local_dir}/${project}
fi


project=iic/speech_fsmn_vad_zh-cn-16k-common-pytorch
if [ ${stage} -eq 1 ]; then
    modelscope download --model $project --local_dir ${local_dir}/${project}
fi
