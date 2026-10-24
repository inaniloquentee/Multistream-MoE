"""Single-GPU correctness/short-training probe, not a scaled comparison."""

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from multistream import Config, initialized_model


def batch(data, rng, size, length, device):
    starts = rng.integers(0, len(data) - length - 1, size=size)
    x = np.stack([data[i:i + length] for i in starts]).astype(np.int64)
    y = np.stack([data[i + 1:i + length + 1] for i in starts]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=list("ABCD"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-experts", type=int, default=4)
    parser.add_argument("--expert-hidden", type=int, default=128)
    parser.add_argument("--dense-hidden", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=65)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This validation requires a BF16 CUDA GPU")
    if args.steps < 2 or args.eval_batches < 1 or args.eval_every < 1:
        raise ValueError("Need >=2 steps, positive eval batches and interval")
    warmup = args.warmup_steps if args.warmup_steps is not None else max(1, args.steps // 10)
    if not 0 <= warmup < args.steps or not 0 < args.min_lr <= args.lr:
        raise ValueError("Invalid learning-rate schedule")
    cfg = Config(vocab_size=args.vocab_size, block_size=args.block_size,
                 n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
                 n_experts=args.n_experts, expert_hidden=args.expert_hidden,
                 dense_hidden=args.dense_hidden)
    train = np.memmap(args.data / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(args.data / "val.bin", dtype=np.uint16, mode="r")
    if min(len(train), len(val)) <= cfg.block_size + 1:
        raise ValueError("Data split shorter than a training sequence")
    model = initialized_model(cfg, args.mode, args.seed).cuda()
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                                   {"params": no_decay, "weight_decay": 0}],
                                  lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
    rng = np.random.default_rng(args.seed)
    validation = [batch(val, np.random.default_rng(2027 + i),
                        args.batch_size, cfg.block_size, "cuda") for i in range(args.eval_batches)]

    def evaluate():
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            losses = [model(x, y)[1].item() for x, y in validation]
        model.train()
        return sum(losses) / len(losses)

    initial_val = evaluate()
    print(json.dumps({"mode": args.mode, "seed": args.seed, "step": 0,
                      "val_nll": initial_val, "train_elapsed_s": 0}), flush=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.monotonic()
    train_elapsed = 0.0
    for step in range(args.steps):
        if step < warmup:
            lr = args.lr * (step + 1) / warmup
        else:
            progress = (step - warmup) / max(1, args.steps - warmup - 1)
            lr = args.min_lr + (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress)) / 2
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = batch(train, rng, args.batch_size, cfg.block_size, "cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss, aux, _ = model(x, y)
            total = loss + 0.01 * aux
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite loss at step {step}")
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            torch.cuda.synchronize()
            train_elapsed += time.monotonic() - start
            current_val = evaluate()
            print(json.dumps({"mode": args.mode, "seed": args.seed, "step": step + 1,
                              "val_nll": current_val, "train_elapsed_s": train_elapsed}), flush=True)
            start = time.monotonic()
    torch.cuda.synchronize()
    elapsed = train_elapsed
    final_val = current_val
    print(json.dumps({"mode": args.mode, "gpu": torch.cuda.get_device_name(0),
                      "torch": torch.__version__, "cuda": torch.version.cuda,
                      "config": asdict(cfg), "seed": args.seed, "steps": args.steps,
                      "lr": args.lr, "min_lr": args.min_lr, "warmup_steps": warmup,
                      "eval_batches": args.eval_batches, "eval_every": args.eval_every,
                      "batch_size": args.batch_size, "params": sum(p.numel() for p in model.parameters()),
                      "initial_val_nll": initial_val, "final_val_nll": final_val,
                      "last_train_nll": loss.item(), "last_grad_norm": norm.item(),
                      "elapsed_s": elapsed, "tokens_per_s": args.steps * args.batch_size
                      * cfg.block_size / elapsed,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated()}, sort_keys=True))


if __name__ == "__main__":
    main()
