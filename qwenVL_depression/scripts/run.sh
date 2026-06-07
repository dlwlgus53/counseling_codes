#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_PROJECT="${WANDB_PROJECT:-mirror-qwenvl}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen3-vl-8b-mirror-kor}"

exec "${SCRIPT_DIR}/finetune_mirror_kor_lora.sh"
