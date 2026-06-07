#!/usr/bin/env bash
set -euo pipefail

set +u
source /home/jihyunlee/anaconda3/etc/profile.d/conda.sh
conda activate collabllm
set -u

SCRIPT=/home/jihyunlee/mirror/MIRROR_code/mirror/translate.py
WORK_DIR=/home/jihyunlee/mirror/translate/train_ko_rewrite_test10

python "$SCRIPT" --work-dir "$WORK_DIR" run \
  --num-rows 10 \
  --poll-seconds 60 \
  --only-translated-rows \
  --output /home/jihyunlee/mirror/MIRROR_code/data/processed/train_ko_rewrite_test10.csv

cat <<EOF

Finished 10-row Korean counseling rewrite test.
Output:
  /home/jihyunlee/mirror/MIRROR_code/data/processed/train_ko_rewrite_test10.csv
EOF
