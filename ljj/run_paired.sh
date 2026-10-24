#!/usr/bin/env bash
# Submit with: sbatch -p h20 --gres=gpu:1 --time=00:40:00 \
#   --output=/home/ljj/Multistream-MoE/ljj/results/slurm-%j.log run_paired.sh
set -euo pipefail

HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export NANOGPT_PATH="${NANOGPT_PATH:-$HERE/_upstream/nanoGPT}"
PYTHON="${PYTHON:-/home/ljj/vllm-pr-59863/.venv/bin/python3}"
DATA="${DATA:-$NANOGPT_PATH/data/shakespeare_char}"
STEPS="${STEPS:-1000}"
mkdir -p "$HERE/results"
cd "$HERE"

"$PYTHON" -m unittest -v test_reference
for seed in 42 43 44; do
  for mode in A B C D; do
    "$PYTHON" train.py --mode "$mode" --seed "$seed" --steps "$STEPS" \
      --data "$DATA" --eval-batches 32 --eval-every 100 \
      | tee "$HERE/results/${mode}-${seed}-${STEPS}.jsonl"
  done
done
