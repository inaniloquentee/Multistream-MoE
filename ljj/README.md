# Experiment workspace

- `multistream_moe/`: reference MoE and Qwen3 adapter.
- `experiments/`: data preparation, checkpoint/precision checks and trainers.
- `tests/`: unit and numerical correctness tests.
- `scripts/`: pinned-framework setup and Slurm launchers.
- `EXPERIMENT.md`: comparison protocol and decision gates.
- `RESULTS.md`: measured evidence and outstanding blockers.

Run all commands from this directory so imports and data paths remain stable:

```bash
PYTHON=${PYTHON:-/home/ljj/vllm-pr-59863/.venv/bin/python3}
bash scripts/bootstrap_transformers.sh
source scripts/use_transformers_fork.sh
$PYTHON -m unittest -v tests.test_qwen
NANOGPT_PATH="$PWD/_upstream/nanoGPT" $PYTHON -m unittest -v tests.test_reference
$PYTHON -m experiments.check_qwen_checkpoint \
  --model cache/qwen3-0.6b-base --data data/qwen-fineweb-proxy
```

For the H20 diagnostic, submit `scripts/run_qwen.sh` from this directory using the command in `RESULTS.md`. Cached weights, token data and raw logs remain local and ignored by Git. Current evidence is correctness-only; the H20 quality comparison remains blocked.
