#!/usr/bin/env bash
# Submit from ljj/ with sbatch -p h20 --gres=gpu:1 --time=06:00:00 \
#   --output=results/qwen-pinned-slurm-%j.log scripts/run_qwen_pinned.sh
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || "${SLURM_JOB_PARTITION:-}" != "h20" ]]; then
  echo "Submit scripts/run_qwen_pinned.sh as an H20 GPU job" >&2
  exit 1
fi
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/home/ljj/vllm-pr-59863/.venv/bin/python3}"
STEPS="${STEPS:-1000}"
DATA="$HERE/data/qwen-fineweb-pinned"
MODEL="$HERE/cache/qwen3-0.6b-base"
cd "$HERE"
mkdir -p results
export PYTHONPATH="$HERE/_deps${PYTHONPATH:+:$PYTHONPATH}"
source scripts/use_transformers_fork.sh
"$PYTHON" -m experiments.prepare_qwen_parquet --verify-only --output "$DATA"
"$PYTHON" -m unittest -v tests.test_qwen tests.test_prepare_qwen_parquet
for seed in 42 43 44; do
  for mode in dense A B C D; do
    "$PYTHON" -m experiments.train_qwen --mode "$mode" --seed "$seed" \
      --steps "$STEPS" --warmup-steps 100 --batch-size 2 --seq-len 128 \
      --eval-batches 128 --eval-every 100 --model "$MODEL" --data "$DATA" \
      --n-experts 4 | tee "results/qwen-pinned-${mode}-${seed}-${STEPS}.jsonl"
  done
done
