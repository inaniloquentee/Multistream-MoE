#!/usr/bin/env bash
# Submit from ljj/ with sbatch -p h20 --gres=gpu:1 --time=04:00:00 \
#   --output=results/qwen-slurm-%j.log scripts/run_qwen.sh
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || "${SLURM_JOB_PARTITION:-}" != "h20" ]]; then
  echo "Submit scripts/run_qwen.sh as a job in the h20 partition" >&2
  exit 1
fi
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/home/ljj/vllm-pr-59863/.venv/bin/python3}"
MODEL="${MODEL:-$HERE/cache/qwen3-0.6b-base}"
DATA="${DATA:-$HERE/data/qwen-fineweb-proxy}"
STEPS="${STEPS:-32}"
cd "$HERE"
mkdir -p results
source "$HERE/scripts/use_transformers_fork.sh"
for mode in dense A B C D; do
  "$PYTHON" -m experiments.train_qwen --mode "$mode" --seed 42 --steps "$STEPS" \
    --model "$MODEL" --data "$DATA" --n-experts 4 \
    | tee "results/qwen-${mode}-42-${STEPS}.jsonl"
done
