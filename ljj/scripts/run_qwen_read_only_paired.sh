#!/usr/bin/env bash
# Submit from ljj/ with SEED=42/43/44 and SELECTION=bias/all.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || "${SLURM_JOB_PARTITION:-}" != "h20" ]]; then
  echo "Submit this script as an H20 GPU job" >&2
  exit 1
fi
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/home/ljj/vllm-pr-59863/.venv/bin/python3}"
SEED="${SEED:?Set SEED to a predeclared training seed}"
SELECTION="${SELECTION:-bias}"
case "$SELECTION" in
  bias) LR=0.001 ;;
  all) LR=0.0001 ;;
  *) echo "SELECTION must be bias or all" >&2; exit 1 ;;
esac
cd "$HERE"
mkdir -p results
export PYTHONPATH="$HERE/_deps${PYTHONPATH:+:$PYTHONPATH}"
source scripts/use_transformers_fork.sh
"$PYTHON" -m experiments.prepare_qwen_parquet --verify-only --output data/qwen-fineweb-pinned
for mode in C D; do
  "$PYTHON" -m experiments.train_qwen --mode "$mode" --seed "$SEED" --steps 1000 \
    --warmup-steps 100 --batch-size 2 --seq-len 128 --eval-batches 128 \
    --eval-every 100 --model cache/qwen3-0.6b-base --data data/qwen-fineweb-pinned \
    --n-experts 4 --initial-stream-contrast 0.2 --fp32-read \
    --train-read-maps-only "$SELECTION" --lr "$LR" \
    | tee "results/qwen-read-only-${SELECTION}-${mode}-${SEED}-1000.jsonl"
done
