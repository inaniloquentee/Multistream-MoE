#!/usr/bin/env bash
# Submit from ljj/ with SEED=42/43/44 after the shared specialist checkpoint exists.
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
"$PYTHON" -m experiments.probe_read_aux_gradients \
  --seed "$SEED" --windows 32 --model cache/qwen3-0.6b-base \
  --data data/qwen-fineweb-pinned --checkpoint "cache/qwen-specialists-$SEED-1000.pt" \
  | tee "results/read-aux-gradient-$SEED.jsonl"
