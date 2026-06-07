#!/usr/bin/env bash
set -euo pipefail

set +u
source /home/jihyunlee/anaconda3/etc/profile.d/conda.sh
conda activate mirror
set -u

SCRIPT=/home/jihyunlee/mirror/MIRROR_code/mirror/translate.py
WORK_DIR=/home/jihyunlee/mirror/translate/train_ko_batch_all

python "$SCRIPT" --work-dir "$WORK_DIR" run \
  --num-rows -1 \
  --poll-seconds 10 \
  --output /home/jihyunlee/mirror/MIRROR_code/data/processed/train_ko.csv

cat <<EOF

Finished full train.csv translation.
Output:
  /home/jihyunlee/mirror/MIRROR_code/data/processed/train_ko.csv
EOF
