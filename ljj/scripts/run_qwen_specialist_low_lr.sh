#!/usr/bin/env bash
# Matched C/D read fitting at 10x lower LR from a common specialist checkpoint.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || "${SLURM_JOB_PARTITION:-}" != "h20" ]]; then
  echo "Submit this script as an H20 GPU job" >&2
  exit 1
fi
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/home/ljj/vllm-pr-59863/.venv/bin/python3}"
SEED="${SEED:?Set SEED to a predeclared training seed}"
cd "$HERE"
mkdir -p results
export PYTHONPATH="$HERE/_deps${PYTHONPATH:+:$PYTHONPATH}"
source scripts/use_transformers_fork.sh
"$PYTHON" -m experiments.prepare_qwen_parquet --verify-only --output data/qwen-fineweb-pinned
checkpoint="cache/qwen-specialists-$SEED-1000.pt"
test -s "$checkpoint"
for mode in C D; do
  "$PYTHON" -m experiments.train_qwen --mode "$mode" --seed "$SEED" --steps 1000 \
    --warmup-steps 100 --batch-size 2 --seq-len 128 --eval-batches 128 \
    --eval-every 100 --model cache/qwen3-0.6b-base --data data/qwen-fineweb-pinned \
    --n-experts 4 --initial-stream-contrast 0.2 --fp32-read \
    --load-specialists "$checkpoint" --train-read-maps-only all --lr 0.00001 \
    --forward-precision fp32 \
    | tee "results/qwen-specialists-low-lr-${mode}-${SEED}-1000.jsonl"
done
