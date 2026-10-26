"""Evaluate whether the warm-start expert-read direction helps held-out NLL."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen3ForCausalLM

from experiments.qwen_diagnostics import (set_initial_expert_read_contrast,
                                          set_initial_stream_contrast)
from experiments.train_qwen import forward_batch, get_batch
from multistream_moe.qwen_reference import QwenMultiStream


def paired(reference, trial):
    differences = [b - a for a, b in zip(reference, trial)]
    mean = sum(differences) / len(differences)
    se = math.sqrt(sum((d - mean) ** 2 for d in differences) /
                   (len(differences) * (len(differences) - 1)))
    return {"delta_nll": mean, "window_se": se}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("H20-class BF16 CUDA device required")
    base = Qwen3ForCausalLM.from_pretrained(str(args.model), dtype=torch.float32,
                                            attn_implementation="sdpa").eval()
    torch.manual_seed(42)
    model = QwenMultiStream(base, "D", n_experts=4).eval()
    set_initial_stream_contrast(model, 0.2)
    set_initial_expert_read_contrast(model, 0.25)
    model.cuda()
    templates = {name: moe.read_bias.detach().clone() / 0.25
                 for name, moe in model.moes.items()}
    val = np.memmap(args.data / "val.bin", mode="r", dtype=np.uint32)
    validation = [get_batch(val, np.random.default_rng(2027 + i), 2, 128, "cuda")
                  for i in range(128)]
    for precision, windows in (("bf16", 128), ("bf16", 32), ("fp32", 32)):
        losses = {}
        for contrast in (0, -0.05, 0.05, -0.25, 0.25):
            with torch.no_grad():
                for name, moe in model.moes.items():
                    moe.read_bias.copy_(contrast * templates[name])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=precision == "bf16"):
                losses[contrast] = [forward_batch(model, "D", x, y)[1].item()
                                    for x, y in validation[:windows]]
            print(json.dumps({"precision": precision, "windows": windows,
                              "read_contrast": contrast,
                              "val_nll": sum(losses[contrast]) / windows,
                              "vs_zero": (paired(losses[0], losses[contrast])
                                          if contrast else None)}), flush=True)


if __name__ == "__main__":
    main()
