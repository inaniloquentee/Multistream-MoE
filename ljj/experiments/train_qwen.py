"""Paired Qwen3-0.6B warm-start probe; not a pretraining claim."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from torch.nn import functional as F
from transformers import Qwen3ForCausalLM

from multistream_moe.qwen_reference import QwenMultiStream


def get_batch(data, rng, batch_size, length, device):
    starts = rng.integers(0, len(data) - length - 1, size=batch_size)
    source = np.stack([data[i:i + length] for i in starts]).astype(np.int64)
    target = np.stack([data[i + 1:i + length + 1] for i in starts]).astype(np.int64)
    return torch.from_numpy(source).to(device), torch.from_numpy(target).to(device)


def forward_batch(model, mode, tokens, targets):
    if mode != "dense":
        return model(tokens, targets)[:3]
    logits = model(tokens).logits
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1))
    return logits, loss, loss.new_zeros(())


def learning_rate(base_lr, step, steps, warmup_steps):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, steps - warmup_steps - 1)
    return base_lr * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)


def mechanism_metrics(model, mode, tokens):
    if mode == "dense":
        return {}
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        streams = model(tokens)[3].float()
    centered = streams - streams.mean(dim=2, keepdim=True)
    second = model.moes["3"]

    def expert_rms(weight):
        return (weight.float() - weight.float().mean(dim=0, keepdim=True)).square().mean().sqrt().item()

    return {"final_stream_centered_rms": centered.square().mean().sqrt().item(),
            "second_read_expert_weight_rms": (expert_rms(second.read_weight)
                                               if mode != "A" else 0.0),
            "second_write_expert_weight_rms": (expert_rms(second.write_weight)
                                                if mode != "A" else 0.0)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dense", *list("ABCD")], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--n-experts", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=0)
    args = parser.parse_args()
    fork_root = os.environ.get("TRANSFORMERS_FORK_ROOT")
    if fork_root and not Path(transformers.__file__).resolve().is_relative_to(Path(fork_root).resolve()):
        raise RuntimeError(f"Transformers imported from {transformers.__file__}, not {fork_root}")
    if (args.steps < 2 or args.eval_batches < 1 or args.eval_every < 1 or
            not 0 <= args.warmup_steps < args.steps):
        raise ValueError("Need at least two steps and positive evaluation settings")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA device required")
    train = np.memmap(args.data / "train.bin", mode="r", dtype=np.uint32)
    val = np.memmap(args.data / "val.bin", mode="r", dtype=np.uint32)
    if min(len(train), len(val)) <= args.seq_len + 1:
        raise ValueError("Data too short for the requested sequence length")
    base = Qwen3ForCausalLM.from_pretrained(str(args.model), dtype=torch.float32,
                                            attn_implementation="sdpa").eval()
    train_rng = np.random.default_rng(args.seed)
    validation = [get_batch(val, np.random.default_rng(2027 + i), args.batch_size,
                            args.seq_len, "cpu") for i in range(args.eval_batches)]
    probe = validation[0][0][:, :8]
    with torch.no_grad():
        original = base(probe).logits
    torch.manual_seed(args.seed)
    model = base if args.mode == "dense" else QwenMultiStream(base, args.mode,
                                                               n_experts=args.n_experts)
    with torch.no_grad():
        lifted = forward_batch(model, args.mode, probe, probe)[0]
    lift_error = (original - lifted).abs().max().item()
    if lift_error > 1e-3:
        raise RuntimeError(f"Qwen warm-start lift mismatch: {lift_error}")
    del original, lifted
    model = model.cuda().train()
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                                   {"params": no_decay, "weight_decay": 0}],
                                  lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
    validation = [(x.cuda(), y.cuda()) for x, y in validation]

    def evaluate():
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            losses = [forward_batch(model, args.mode, x, y)[1].item() for x, y in validation]
        model.train()
        return sum(losses) / len(losses)

    initial = evaluate()
    print(json.dumps({"mode": args.mode, "seed": args.seed, "step": 0,
                      "val_nll": initial, "lift_max_abs": lift_error,
                      "params": sum(p.numel() for p in model.parameters()),
                      "gpu": torch.cuda.get_device_name(0),
                      "transformers_source": str(Path(transformers.__file__).resolve()),
                      "transformers_version": transformers.__version__}), flush=True)
    torch.cuda.reset_peak_memory_stats()
    train_elapsed = 0.0
    start = time.monotonic()
    for step in range(args.steps):
        lr = learning_rate(args.lr, step, args.steps, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = get_batch(train, train_rng, args.batch_size, args.seq_len, "cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss, aux = forward_batch(model, args.mode, x, y)
            total = loss + 0.01 * aux
        if not torch.isfinite(total):
            raise RuntimeError(f"Non-finite loss at step {step}")
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            torch.cuda.synchronize()
            train_elapsed += time.monotonic() - start
            current = evaluate()
            record = {"mode": args.mode, "seed": args.seed, "step": step + 1,
                              "val_nll": current, "train_elapsed_s": train_elapsed,
                              "last_train_nll": loss.item(), "last_grad_norm": norm.item(),
                              "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
            if step + 1 == args.steps:
                record["mechanism"] = mechanism_metrics(model, args.mode, validation[0][0])
            print(json.dumps(record), flush=True)
            start = time.monotonic()


if __name__ == "__main__":
    main()
