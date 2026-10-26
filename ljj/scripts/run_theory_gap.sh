#!/usr/bin/env bash
# Submit from ljj/: sbatch -p h20 --gres=gpu:1 --time=00:30:00 --output="$PWD/results/theory-gap-slurm-%j.log" scripts/run_theory_gap.sh
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
"$PYTHON" -m unittest -v tests.test_theory_gap
"$PYTHON" -m experiments.check_theory_gap --model cache/qwen3-0.6b-base \
  --data data/qwen-fineweb-pinned | tee results/theory-gap.jsonl
