#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -f /home/jihyunlee/anaconda3/etc/profile.d/conda.sh ]]; then
  source /home/jihyunlee/anaconda3/etc/profile.d/conda.sh
  conda activate mirror
elif [[ -f /home/jihyunlee/anaconda3/bin/activate ]]; then
  source /home/jihyunlee/anaconda3/bin/activate mirror
fi

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

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

GPT_MODEL="${GPT_MODEL:-gpt-4o}"

cd "${ROOT_DIR}"

echo "Building PTSD add_dataset retrieval artifacts"
"${PYTHON_BIN}" src/build_add_dataset_example.py \
  --profile-model "${GPT_MODEL}" \
  --dialogue-model "${GPT_MODEL}" \
  "$@"

echo "Done."
