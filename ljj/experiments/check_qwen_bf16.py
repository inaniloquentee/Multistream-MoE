"""One-batch full-checkpoint precision/gradient probe, not a training result."""

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import Qwen3ForCausalLM

from multistream_moe.qwen_reference import QwenMultiStream
from experiments.train_qwen import forward_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--mode", choices=["dense", *list("ABCD")], default="D")
    args = parser.parse_args()
    fork_root = os.environ.get("TRANSFORMERS_FORK_ROOT")
    if fork_root and not Path(transformers.__file__).resolve().is_relative_to(Path(fork_root).resolve()):
        raise RuntimeError(f"Transformers imported from {transformers.__file__}, not {fork_root}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA device required")
    val = np.memmap(args.data / "val.bin", dtype=np.uint32, mode="r")
    if args.seq_len < 2 or len(val) <= args.seq_len:
        raise ValueError("Need at least two tokens and a longer validation split")
    tokens = torch.tensor(val[:args.seq_len].astype(np.int64), device="cuda")[None, :]
    targets = torch.tensor(val[1:args.seq_len + 1].astype(np.int64), device="cuda")[None, :]

    base = Qwen3ForCausalLM.from_pretrained(str(args.model), dtype=torch.float32,
                                            attn_implementation="sdpa").eval()
    if args.mode == "dense":
        model = base.cuda()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            logits, loss, _ = forward_batch(model, "dense", tokens, targets)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mixed_logits, mixed_loss, _ = forward_batch(model, "dense", tokens, targets)
        print(json.dumps({
            "gpu": torch.cuda.get_device_name(0),
            "transformers_source": str(Path(transformers.__file__).resolve()),
            "mode": "dense", "seq_len": args.seq_len,
            "fp32_nll": loss.item(), "bf16_nll": mixed_loss.item(),
            "max_logit_abs_error": (logits - mixed_logits.float()).abs().max().item(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }, sort_keys=True), flush=True)
        return
    torch.manual_seed(42)
    model = QwenMultiStream(base, args.mode, n_experts=4).eval()
    if args.mode != "D":
        model = model.cuda()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            logits, loss, _, _ = model(tokens, targets)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mixed_logits, mixed_loss, _, _ = model(tokens, targets)
        print(json.dumps({
            "gpu": torch.cuda.get_device_name(0),
            "transformers_source": str(Path(transformers.__file__).resolve()),
            "mode": args.mode, "seq_len": args.seq_len,
            "fp32_nll": loss.item(), "bf16_nll": mixed_loss.item(),
            "max_logit_abs_error": (logits - mixed_logits.float()).abs().max().item(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }, sort_keys=True), flush=True)
        return
    model.requires_grad_(False)
    selected = {
        "first_write": model.moes["1"].write_weight,
        "second_read": model.moes["3"].read_weight,
    }
    for parameter in selected.values():
        parameter.requires_grad_(True)
    model = model.cuda()
    del base
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    logits, loss, _, _ = model(tokens, targets)
    loss.backward()
    full_grad = {name: parameter.grad.detach().clone() for name, parameter in selected.items()}
    full_loss = loss.item()
    full_logits = logits.detach()
    model.zero_grad(set_to_none=True)
    del logits, loss

    with torch.autocast("cuda", dtype=torch.bfloat16):
        mixed_logits, mixed_loss, _, _ = model(tokens, targets)
    mixed_loss.backward()
    gradients = {}
    for name, parameter in selected.items():
        full = full_grad[name]
        mixed = parameter.grad.detach()
        if not torch.isfinite(full).all() or not torch.isfinite(mixed).all():
            raise RuntimeError(f"Non-finite {name} gradient")
        gradients[name] = {
            "fp32_norm": full.norm().item(),
            "bf16_norm": mixed.norm().item(),
            "difference_norm": (full - mixed).norm().item(),
        }
    if not all(item["fp32_norm"] > 0 and item["bf16_norm"] > 0 for item in gradients.values()):
        raise RuntimeError("A selected read/write gradient vanished")
    if not torch.isfinite(mixed_loss):
        raise RuntimeError("Non-finite BF16 loss")
    torch.cuda.synchronize()
    print(json.dumps({
        "gpu": torch.cuda.get_device_name(0),
        "transformers_source": str(Path(transformers.__file__).resolve()),
        "mode": "D", "seq_len": args.seq_len,
        "fp32_nll": full_loss, "bf16_nll": mixed_loss.item(),
        "max_logit_abs_error": (full_logits - mixed_logits.float()).abs().max().item(),
        "gradients": gradients,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
