#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADD_DATASET_DIR="${ROOT_DIR}/data/add_dataset"

if [[ -f /home/jihyunlee/anaconda3/etc/profile.d/conda.sh ]]; then
  source /home/jihyunlee/anaconda3/etc/profile.d/conda.sh
  conda activate mirror
elif [[ -f /home/jihyunlee/anaconda3/bin/activate ]]; then
  source /home/jihyunlee/anaconda3/bin/activate mirror
fi

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
CONDA_ENV_DIR="$(dirname "$(dirname "${PYTHON_BIN}")")"
CUDA13_LIB_DIR="${CONDA_ENV_DIR}/lib/python3.10/site-packages/nvidia/cu13/lib"
if [[ -d "${CUDA13_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${CUDA13_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

API_CONFIG="${API_CONFIG:-/home/jihyunlee/mirror/MIRROR_code/configs/api.json}"
if [[ -z "${OPENAI_API_KEY:-}" && -f "${API_CONFIG}" ]]; then
  OPENAI_API_KEY="$("${PYTHON_BIN}" - "${API_CONFIG}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open(encoding="utf-8") as f:
    cfg = json.load(f)
print(cfg.get("api-key") or cfg.get("api-key-personal") or "")
PY
)"
  export OPENAI_API_KEY
fi

ADD_CSV="${ADD_CSV:-${ROOT_DIR}/data/raw/add_dataset.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-${ROOT_DIR}/data/mirror_images}"
QWEN_JSONL="${QWEN_JSONL:-${ROOT_DIR}/data/add_dataset_qwen.jsonl}"
CONTEXT_LEN="${CONTEXT_LEN:-10}"
SEED="${SEED:-19}"
GPT_MODEL="${GPT_MODEL:-gpt-4o}"

SKIP_IMAGE_SYNTHESIS="${SKIP_IMAGE_SYNTHESIS:-0}"
SKIP_GPT_IMAGE_ANNOTATION="${SKIP_GPT_IMAGE_ANNOTATION:-0}"
SKIP_PHOTOMAKER="${SKIP_PHOTOMAKER:-0}"
BUILD_QWEN_JSONL="${BUILD_QWEN_JSONL:-0}"

cd "${ROOT_DIR}"

echo "[1/2] Building add_dataset CSV: ${ADD_CSV}"
"${PYTHON_BIN}" src/build_add_dataset_example.py \
  --output-csv "${ADD_CSV}" \
  --profile-model "${GPT_MODEL}" \
  --dialogue-model "${GPT_MODEL}" \
  "$@"

if [[ "${SKIP_IMAGE_SYNTHESIS}" != "1" ]]; then
  echo "[2/2] Generating image prompts/images into: ${IMAGE_ROOT}"
  IMAGE_SYNTHESIS_ARGS=(
    --data-path "${ADD_CSV}"
    --save-dir "${IMAGE_ROOT}"
    --gpt-model "${GPT_MODEL}"
    --num-steps 1
    --seed "${SEED}"
  )
  if [[ "${SKIP_GPT_IMAGE_ANNOTATION}" == "1" ]]; then
    IMAGE_SYNTHESIS_ARGS+=(--skip-gpt)
  fi
  if [[ "${SKIP_PHOTOMAKER}" == "1" ]]; then
    IMAGE_SYNTHESIS_ARGS+=(--skip-photomaker)
  fi
  "${PYTHON_BIN}" data/add_dataset/image_synthesis/run_image_synthesis.py "${IMAGE_SYNTHESIS_ARGS[@]}"
else
  echo "[2/2] Skipping image synthesis"
fi

if [[ "${BUILD_QWEN_JSONL}" == "1" ]]; then
  echo "[optional] Building Qwen JSONL: ${QWEN_JSONL}"
  "${PYTHON_BIN}" src/build_mirror_ko_qwen_json.py \
    --csv "${ADD_CSV}" \
    --image-root "${IMAGE_ROOT}" \
    --output "${QWEN_JSONL}" \
    --context-len "${CONTEXT_LEN}"
fi

echo "Done."
