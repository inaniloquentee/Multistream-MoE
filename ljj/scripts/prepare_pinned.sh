#!/usr/bin/env bash
# Submit from ljj/ with sbatch -p h20 --gres=gpu:0 --cpus-per-task=4 \
#   --mem=24G --time=02:00:00 --output=results/pinned-%j.log scripts/prepare_pinned.sh
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || "${SLURM_JOB_PARTITION:-}" != "h20" ]]; then
  echo "Submit scripts/prepare_pinned.sh to the h20 partition" >&2
  exit 1
fi
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/home/ljj/vllm-pr-59863/.venv/bin/python3}"
cd "$HERE"
mkdir -p cache results
export HF_HOME="$HERE/cache"
export PYTHONPATH="$HERE/_deps${PYTHONPATH:+:$PYTHONPATH}"
source scripts/use_transformers_fork.sh
bash scripts/bootstrap_parquet.sh
"$PYTHON" -m unittest -v tests.test_prepare_qwen_parquet
"$PYTHON" -m experiments.prepare_qwen_parquet \
  --train-tokens 20000000 --val-tokens 2000000
