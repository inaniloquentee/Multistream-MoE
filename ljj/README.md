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

For the H20 diagnostics, submit `scripts/run_qwen.sh` (32 steps) or
`scripts/run_qwen_paired.sh` (1,000 steps, three seeds) from this directory.
Measured results and the raw JSONL logs are in `RESULTS.md` and `evidence/`.
Cached weights and token data remain local and ignored by Git. The proxy-data
diagnostics do not establish a quality gain; fixed-snapshot Q2 and EP are open.

The pinned-data diagnostic uses `scripts/prepare_pinned.sh` as a CPU-only Slurm
job, then `scripts/run_qwen_pinned.sh` as a single-H20 job. Preparation downloads
one exact FineWeb-Edu Parquet revision with Hugging Face Hub, uses a locally
installed PyArrow wheel in `_deps/`, and checks the source, tokenizer, split IDs
and token binaries before training. The exact protocol is in `EXPERIMENT.md`.
