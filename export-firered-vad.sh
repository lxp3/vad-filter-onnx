#! /bin/bash

model_dir=/appsvc/lxp/downloads/models/FireRedTeam/FireRedVAD/Stream-VAD
onnx_path=public/models/firered_vad.onnx

PYTHONPATH=/appsvc/lxp/repos/FireRedVAD:${PYTHONPATH} \
python3 scripts/export_onnx_firered_vad.py \
  --model-dir ${model_dir}  \
  --onnx-path ${onnx_path}
