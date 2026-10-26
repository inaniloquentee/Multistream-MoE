#!/usr/bin/env bash
# Submit from ljj/: sbatch -p h20 --gres=gpu:1 --time=01:00:00 --output="$PWD/results/qwen-fp32-read-slurm-%j.log" scripts/run_qwen_fp32_read_paired.sh
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || "${SLURM_JOB_PARTITION:-}" != "h20" ]]; then
  echo "Submit this script as an H20 GPU job" >&2
  exit 1
fi
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/home/ljj/vllm-pr-59863/.venv/bin/python3}"
cd "$HERE"
mkdir -p results
export PYTHONPATH="$HERE/_deps${PYTHONPATH:+:$PYTHONPATH}"
source scripts/use_transformers_fork.sh
"$PYTHON" -m experiments.prepare_qwen_parquet --verify-only --output data/qwen-fineweb-pinned
"$PYTHON" -m unittest -v tests.test_read_precision tests.test_qwen tests.test_qwen_diagnostics
for mode in C D; do
  diagnostic=()
  if [[ "$mode" == D ]]; then diagnostic=(--diagnose --diagnostic-windows 32); fi
  "$PYTHON" -m experiments.train_qwen --mode "$mode" --seed 42 --steps 1000 \
    --warmup-steps 100 --batch-size 2 --seq-len 128 --eval-batches 128 \
    --eval-every 100 --model cache/qwen3-0.6b-base --data data/qwen-fineweb-pinned \
    --n-experts 4 --initial-stream-contrast 0.2 --fp32-read "${diagnostic[@]}" \
    | tee "results/qwen-fp32-read-${mode}-42-1000.jsonl"
done
