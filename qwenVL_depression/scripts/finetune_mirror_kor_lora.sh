#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
CSV_PATH="${CSV_PATH:-${ROOT_DIR}/data/raw/train_ko.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-${ROOT_DIR}/data/mirror_images}"
DATA_PATH="${DATA_PATH:-${ROOT_DIR}/data/mirror_ko_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/checkpoints/qwen3-vl-8b-mirror-kor-lora}"

MAX_LENGTH="${MAX_LENGTH:-4096}"
CONTEXT_LEN="${CONTEXT_LEN:-10}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:--1}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
SAVE_STEPS="${SAVE_STEPS:-500}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
IO_LOG_STEPS="${IO_LOG_STEPS:-50}"
IO_LOG_NUM_SAMPLES="${IO_LOG_NUM_SAMPLES:-1}"
IO_LOG_MAX_NEW_TOKENS="${IO_LOG_MAX_NEW_TOKENS:-128}"
IO_LOG_GENERATE="${IO_LOG_GENERATE:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29501}"
WANDB_PROJECT="${WANDB_PROJECT:-mirror-qwenvl}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen3-vl-8b-mirror-kor-lora}"
REPORT_TO="${REPORT_TO:-wandb}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
  MIRROR_ENV_PYTHON="/home/jihyunlee/anaconda3/envs/mirror/bin/python"
  if [[ "${PYTHON_BIN}" != *"/envs/mirror/"* && -x "${MIRROR_ENV_PYTHON}" ]]; then
    PYTHON_BIN="${MIRROR_ENV_PYTHON}"
  fi
fi
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "${PYTHON_BIN}")/torchrun}"

export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT
export WANDB_RUN_NAME

CONDA_ENV_DIR="$(dirname "$(dirname "${PYTHON_BIN}")")"
CUDA13_LIB_DIR="${CONDA_ENV_DIR}/lib/python3.10/site-packages/nvidia/cu13/lib"
if [[ -d "${CUDA13_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${CUDA13_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

"${PYTHON_BIN}" "${ROOT_DIR}/src/build_mirror_ko_qwen_json.py" \
  --csv "${CSV_PATH}" \
  --image-root "${IMAGE_ROOT}" \
  --output "${DATA_PATH}" \
  --context-len "${CONTEXT_LEN}"

EFFECTIVE_BATCH_SIZE=$((PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS * NPROC_PER_NODE))
mkdir -p "${OUTPUT_DIR}"
cat > "${OUTPUT_DIR}/launch_args.env" <<EOF
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
CSV_PATH=${CSV_PATH}
IMAGE_ROOT=${IMAGE_ROOT}
DATA_PATH=${DATA_PATH}
OUTPUT_DIR=${OUTPUT_DIR}
MAX_LENGTH=${MAX_LENGTH}
CONTEXT_LEN=${CONTEXT_LEN}
EPOCHS=${EPOCHS}
MAX_STEPS=${MAX_STEPS}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}
NPROC_PER_NODE=${NPROC_PER_NODE}
EFFECTIVE_BATCH_SIZE=${EFFECTIVE_BATCH_SIZE}
LEARNING_RATE=${LEARNING_RATE}
LORA_R=${LORA_R}
LORA_ALPHA=${LORA_ALPHA}
LORA_DROPOUT=${LORA_DROPOUT}
SAVE_STEPS=${SAVE_STEPS}
LOGGING_STEPS=${LOGGING_STEPS}
IO_LOG_STEPS=${IO_LOG_STEPS}
IO_LOG_NUM_SAMPLES=${IO_LOG_NUM_SAMPLES}
IO_LOG_MAX_NEW_TOKENS=${IO_LOG_MAX_NEW_TOKENS}
IO_LOG_GENERATE=${IO_LOG_GENERATE}
WANDB_PROJECT=${WANDB_PROJECT}
WANDB_RUN_NAME=${WANDB_RUN_NAME}
REPORT_TO=${REPORT_TO}
PYTHON_BIN=${PYTHON_BIN}
TORCHRUN_BIN=${TORCHRUN_BIN}
EOF

echo "GPUs / nproc_per_node: ${NPROC_PER_NODE}"
echo "Per-device batch size: ${PER_DEVICE_BATCH_SIZE}"
echo "Gradient accumulation steps: ${GRAD_ACCUM_STEPS}"
echo "Effective train batch size: ${PER_DEVICE_BATCH_SIZE} * ${GRAD_ACCUM_STEPS} * ${NPROC_PER_NODE} = ${EFFECTIVE_BATCH_SIZE}"
echo "IO sample logging every ${IO_LOG_STEPS} optimizer steps; samples=${IO_LOG_NUM_SAMPLES}; generate=${IO_LOG_GENERATE}"
echo "Saved launch args: ${OUTPUT_DIR}/launch_args.env"

IO_GENERATE_ARGS=()
if [[ "${IO_LOG_GENERATE}" == "0" ]]; then
  IO_GENERATE_ARGS+=(--io_log_no_generate)
fi

TORCH_LAUNCH=()
if [[ -x "${TORCHRUN_BIN}" ]]; then
  TORCH_LAUNCH=("${TORCHRUN_BIN}")
else
  TORCH_LAUNCH=("${PYTHON_BIN}" -m torch.distributed.run)
fi

"${TORCH_LAUNCH[@]}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  "${ROOT_DIR}/src/train_mirror_kor_lora.py" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --data_path "${DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_length "${MAX_LENGTH}" \
  --num_train_epochs "${EPOCHS}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --save_steps "${SAVE_STEPS}" \
  --logging_steps "${LOGGING_STEPS}" \
  --io_log_steps "${IO_LOG_STEPS}" \
  --io_log_num_samples "${IO_LOG_NUM_SAMPLES}" \
  --io_log_max_new_tokens "${IO_LOG_MAX_NEW_TOKENS}" \
  "${IO_GENERATE_ARGS[@]}" \
  --run_name "${WANDB_RUN_NAME}" \
  --report_to "${REPORT_TO}" \
  --bf16 \
  --gradient_checkpointing \
  --load_in_4bit
