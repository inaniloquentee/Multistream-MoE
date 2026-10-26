#!/usr/bin/env bash
# Submit from ljj/ with sbatch -p h20 --gres=gpu:1 --time=04:00:00 \
#   --output=results/qwen-paired-slurm-%j.log scripts/run_qwen_paired.sh
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || "${SLURM_JOB_PARTITION:-}" != "h20" ]]; then
  echo "Submit scripts/run_qwen_paired.sh as a job in the h20 partition" >&2
  exit 1
fi

HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/home/ljj/vllm-pr-59863/.venv/bin/python3}"
MODEL="${MODEL:-$HERE/cache/qwen3-0.6b-base}"
DATA="${DATA:-$HERE/data/qwen-fineweb-proxy}"
STEPS="${STEPS:-1000}"
EVAL_BATCHES="${EVAL_BATCHES:-64}"
EVAL_EVERY="${EVAL_EVERY:-100}"
cd "$HERE"
mkdir -p results
source scripts/use_transformers_fork.sh
sha256sum "$DATA/train.bin" "$DATA/val.bin"
"$PYTHON" -m unittest -v tests.test_qwen
for seed in 42 43 44; do
  for mode in dense A B C D; do
    "$PYTHON" -m experiments.train_qwen --mode "$mode" --seed "$seed" \
      --steps "$STEPS" --eval-batches "$EVAL_BATCHES" --eval-every "$EVAL_EVERY" \
      --model "$MODEL" --data "$DATA" --n-experts 4 \
      | tee "results/qwen-${mode}-${seed}-${STEPS}-paired.jsonl"
  done
done
