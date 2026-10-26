"""Isolate FP32 read accumulation inside otherwise BF16 Qwen forward passes."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import Qwen3ForCausalLM

from experiments.probe_target_alignment import (downhill_direction, gradients,
                                                losses, paired)
from experiments.qwen_diagnostics import activation_probe, set_initial_stream_contrast
from experiments.train_qwen import get_batch
from multistream_moe.qwen_reference import QwenMultiStream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    args = parser.parse_args()
    fork_root = os.environ.get("TRANSFORMERS_FORK_ROOT")
    if fork_root and not Path(transformers.__file__).resolve().is_relative_to(Path(fork_root).resolve()):
        raise RuntimeError("Transformers fork is not active")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA device required")
    base = Qwen3ForCausalLM.from_pretrained(str(args.model), dtype=torch.float32,
                                            attn_implementation="sdpa").eval()
    torch.manual_seed(42)
    model = QwenMultiStream(base, "D", n_experts=4).eval()
    set_initial_stream_contrast(model, 0.2)
    model.requires_grad_(False)
    for moe in model.moes.values():
        moe.read_bias.requires_grad_(True)
    model.cuda()
    train = np.memmap(args.data / "train.bin", mode="r", dtype=np.uint32)
    val = np.memmap(args.data / "val.bin", mode="r", dtype=np.uint32)
    validation = [get_batch(val, np.random.default_rng(2027 + i), 2, 128, "cuda")
                  for i in range(128)]
    print(json.dumps({"gpu": torch.cuda.get_device_name(0),
                      "transformers_source": str(Path(transformers.__file__).resolve()),
                      "train_gradient_windows": 32, "validation_windows": 128,
                      "amplitude": 0.05, "direction_seeds": [42, 43, 44]}), flush=True)
    directions = {}
    for seed in (42, 43, 44):
        rng = np.random.default_rng(35000 + seed)
        batches = [get_batch(train, rng, 2, 128, "cuda") for _ in range(32)]
        directions[seed] = downhill_direction(gradients(model, batches))
    original = {name: moe.read_bias.detach().clone() for name, moe in model.moes.items()}
    for variant, precision, fp32_read in (("bf16", "bf16", False),
                                          ("bf16_fp32_read", "bf16", True),
                                          ("fp32", "fp32", False)):
        for moe in model.moes.values():
            moe.fp32_read = fp32_read
        baseline = losses(model, validation, precision)
        for seed, direction in directions.items():
            try:
                with torch.no_grad():
                    for name, moe in model.moes.items():
                        moe.read_bias.copy_(original[name] + 0.05 * direction[name])
                trial = losses(model, validation, precision)
                activation = (activation_probe(model, validation[0][0], precision)["3"]
                              if seed == 42 else None)
            finally:
                with torch.no_grad():
                    for name, moe in model.moes.items():
                        moe.read_bias.copy_(original[name])
            print(json.dumps({"variant": variant, "seed": seed, "windows": 128,
                              "baseline_nll": sum(baseline) / len(baseline),
                              "trial_nll": sum(trial) / len(trial),
                              "paired": paired(baseline, trial),
                              "second_moe_activation": activation}), flush=True)


if __name__ == "__main__":
    main()
