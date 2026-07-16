#! /bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXPORT_PY="${REPO_ROOT}/scripts/export_onnx_fsmn_vad.py"
OUTPUT_DIR="${REPO_ROOT}/public/models"
BASE_MODEL_DIR="/data/user/lxp/vad/1-20250617/iic"

export PYTHONPATH="/data/user/lxp/projects/vad-filter-onnx/debug/FunASR:${REPO_ROOT}/scripts:${PYTHONPATH:-}"

if [[ ! -f "${EXPORT_PY}" ]]; then
  echo "export script not found: ${EXPORT_PY}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

export_one() {
  local model_name="$1"
  local sample_rate="$2"
  local model_dir="${BASE_MODEL_DIR}/${model_name}"

  if [[ ! -d "${model_dir}" ]]; then
    echo "model dir not found: ${model_dir}" >&2
    exit 1
  fi

  echo "==== Exporting ${model_name} (sample_rate=${sample_rate}) ===="
  python3 "${EXPORT_PY}" \
    --model-dir "${model_dir}" \
    --sample-rate "${sample_rate}" \
    --output-dir "${OUTPUT_DIR}" \
    --quantize 1
}

export_one "speech_fsmn_vad_zh-cn-8k-common" 8000
export_one "speech_fsmn_vad_zh-cn-16k-common-pytorch" 16000

echo "Done. Models written to ${OUTPUT_DIR}:"
ls -lh "${OUTPUT_DIR}"/fsmn_vad.*.onnx
