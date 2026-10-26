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
and token binaries before training. The three-seed, 1,000-step pinned diagnostic
completed without a stable D quality gain; see `RESULTS.md` and the raw logs in
`evidence/`. The full 10,000-step, full-validation comparison and EP remain open.

For causal read-map and BF16 activation diagnostics, submit
`scripts/run_qwen_diagnostics.sh`; for the paired C/D stream-contrast intervention,
submit `scripts/run_qwen_contrast.sh`. Both are H20 Slurm jobs using the same
pinned data. The measured mechanism results and raw JSONL are in `MECHANISM.md`
and `evidence/h20-2026-10-27/mechanism/`. These single-seed probes do not replace
the preregistered longer paired comparison.

The first theory-to-warm-start mismatch is isolated with
`scripts/run_theory_gap.sh`; see `FIRST_GAP.md` and its H20 forward-only log in
`evidence/h20-2026-10-27/theory-gap/`.

The optional algorithm-level read initialization and the next gap between
effective expert reads and held-out quality are tested with
`scripts/run_qwen_read_init.sh` and `scripts/run_initial_read_direction.sh`;
see `SECOND_GAP.md` and `evidence/h20-2026-10-27/read-init/`.

Train-only gradient directions, full-window precision isolation, and a paired
C/D FP32-read training probe are in `TARGET_ALIGNMENT.md`; run
`scripts/run_target_alignment.sh`, `scripts/run_read_precision.sh`, or
`scripts/run_qwen_fp32_read_paired.sh`. Raw H20 logs are under
`evidence/h20-2026-10-27/target-alignment/`.

Frozen-trunk C/D read-map comparisons, a common routed-expert preparation
stage, FP32 and auxiliary-loss ablations, and larger-sample checks are in
`READ_ONLY.md`. Run the paired scripts under `scripts/run_qwen_read_only_*`
or `scripts/run_qwen_specialist_*`; the raw logs are in
`evidence/h20-2026-10-27/read-only/` and `specialist-read/`. The larger-sample
paired diagnostic did not establish a reliable D-over-C quality gain.
