"""Separate next-token and load-balancing gradients after specialist training."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen3ForCausalLM

from experiments.qwen_diagnostics import set_initial_stream_contrast
from experiments.train_qwen import (forward_batch, freeze_except_read_maps,
                                    get_batch, load_specialists)
from multistream_moe.qwen_reference import QwenMultiStream


def gradients(model, batches, component):
    model.zero_grad(set_to_none=True)
    for tokens, targets in batches:
        _, nll, aux = forward_batch(model, "D", tokens, targets)
        ((nll if component == "nll" else aux) / len(batches)).backward()
    result = {name: param.grad.detach().clone() if param.grad is not None
              else torch.zeros_like(param)
              for name, param in model.named_parameters() if param.requires_grad}
    model.zero_grad(set_to_none=True)
    return result


def component(gradient, independent):
    return {name: (tensor - tensor.mean(dim=0, keepdim=True) if independent else
                   tensor.mean(dim=0, keepdim=True).expand_as(tensor))
            for name, tensor in gradient.items()}


def dot(left, right):
    return sum((left[name] * right[name]).sum().item() for name in left)


def cosine(left, right):
    norm = math.sqrt(dot(left, left) * dot(right, right))
    return dot(left, right) / norm if norm else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--windows", type=int, default=32)
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.windows < 2:
        raise ValueError("CUDA and at least two windows required")
    base = Qwen3ForCausalLM.from_pretrained(str(args.model), dtype=torch.float32,
                                            attn_implementation="sdpa").eval()
    torch.manual_seed(args.seed)
    model = QwenMultiStream(base, "D", n_experts=4).eval()
    set_initial_stream_contrast(model, 0.2)
    load_specialists(model, torch.load(args.checkpoint, map_location="cpu", weights_only=True), 0.2)
    for moe in model.moes.values():
        moe.fp32_read = True
    freeze_except_read_maps(model, "all")
    model.cuda()
    train = np.memmap(args.data / "train.bin", mode="r", dtype=np.uint32)
    val = np.memmap(args.data / "val.bin", mode="r", dtype=np.uint32)
    train_rng = np.random.default_rng(42000 + args.seed)
    train_batches = [get_batch(train, train_rng, 2, 128, "cuda") for _ in range(args.windows)]
    val_batches = [get_batch(val, np.random.default_rng(2027 + i), 2, 128, "cuda")
                   for i in range(args.windows)]
    train_nll = gradients(model, train_batches, "nll")
    train_aux = gradients(model, train_batches, "aux")
    val_nll = gradients(model, val_batches, "nll")
    for independent in (False, True):
        a = component(train_nll, independent)
        b = component(train_aux, independent)
        c = component(val_nll, independent)
        combined = {name: a[name] + 0.01 * b[name] for name in a}
        print(json.dumps({"seed": args.seed, "windows": args.windows,
                          "subspace": "expert_independent" if independent else "shared",
                          "train_nll_grad_norm": math.sqrt(dot(a, a)),
                          "weighted_aux_grad_norm": 0.01 * math.sqrt(dot(b, b)),
                          "train_nll_val_nll_cosine": cosine(a, c),
                          "train_total_val_nll_cosine": cosine(combined, c),
                          "train_aux_val_nll_cosine": cosine(b, c)}), flush=True)


if __name__ == "__main__":
    main()
