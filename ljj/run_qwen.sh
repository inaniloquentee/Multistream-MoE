#!/usr/bin/env bash
# Submit from ljj/ with sbatch -p h20 --gres=gpu:1 --time=04:00:00 \
#   --output=results/qwen-slurm-%j.log run_qwen.sh
set -euo pipefail
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${PYTHON:-/home/ljj/vllm-pr-59863/.venv/bin/python3}"
MODEL="${MODEL:-$HERE/cache/qwen3-0.6b-base}"
DATA="${DATA:-$HERE/data/qwen-fineweb-proxy}"
STEPS="${STEPS:-32}"
cd "$HERE"
mkdir -p results
source "$HERE/use_transformers_fork.sh"
for mode in dense A B C D; do
  "$PYTHON" train_qwen.py --mode "$mode" --seed 42 --steps "$STEPS" \
    --model "$MODEL" --data "$DATA" --n-experts 4 \
    | tee "results/qwen-${mode}-42-${STEPS}.jsonl"
done
