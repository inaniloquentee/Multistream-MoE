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
from experiments.qwen_diagnostics import (diagnose, gradient_probe,
                                          set_initial_expert_read_contrast,
                                          set_initial_stream_contrast)


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


def mechanism_metrics(model, mode, tokens, precision="bf16"):
    if mode == "dense":
        return {}
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=precision == "bf16"):
        streams = model(tokens)[3].float()
    centered = streams - streams.mean(dim=2, keepdim=True)
    second = model.moes["3"]

    def expert_rms(weight):
        return (weight.float() - weight.float().mean(dim=0, keepdim=True)).square().mean().sqrt().item()

    return {"final_stream_centered_rms": centered.square().mean().sqrt().item(),
            "second_read_expert_weight_rms": (expert_rms(second.read_weight)
                                               if mode != "A" else 0.0),
            "second_read_expert_bias_rms": (expert_rms(second.read_bias)
                                             if mode != "A" else 0.0),
            "second_write_expert_weight_rms": (expert_rms(second.write_weight)
                                                if mode != "A" else 0.0)}


def freeze_except_read_maps(model, selection):
    """Keep the Qwen trunk, experts, router, mixing, and writes fixed."""
    if selection not in ("bias", "all") or getattr(model, "mode", None) not in ("C", "D"):
        raise ValueError("Read-only training requires C/D and bias or all")
    model.requires_grad_(False)
    for moe in model.moes.values():
        moe.read_bias.requires_grad_(True)
        if selection == "all":
            moe.read_weight.requires_grad_(True)


def specialist_parameters(model):
    """Only routed experts and their routers; the shared expert remains frozen."""
    if getattr(model, "mode", None) not in ("C", "D"):
        raise ValueError("Specialist training requires C/D")
    selected = {}
    for layer, moe in model.moes.items():
        routed = tuple(f"experts.{j}." for j in range(1, moe.n_experts + 1))
        for name, param in moe.named_parameters():
            if name.startswith("router.") or name.startswith(routed):
                selected[f"moes.{layer}.{name}"] = param
    return selected


def freeze_except_specialists(model):
    selected = specialist_parameters(model)
    model.requires_grad_(False)
    for param in selected.values():
        param.requires_grad_(True)


def load_specialists(model, checkpoint, stream_contrast):
    selected = specialist_parameters(model)
    if (checkpoint.get("source_mode") != "C" or
            checkpoint.get("initial_stream_contrast") != stream_contrast or
            set(checkpoint.get("parameters", {})) != set(selected)):
        raise ValueError("Specialist checkpoint has a mismatched C architecture or stream contrast")
    with torch.no_grad():
        for name, param in selected.items():
            tensor = checkpoint["parameters"][name]
            if tensor.shape != param.shape or tensor.dtype != param.dtype:
                raise ValueError(f"Specialist checkpoint shape/dtype mismatch: {name}")
            param.copy_(tensor)


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
    parser.add_argument("--aux-weight", type=float, default=0.01,
                        help="Router load-balancing term; use zero for an NLL-only read-map ablation")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--diagnose", action="store_true",
                        help="Measure D read/write counterfactuals on fixed validation windows")
    parser.add_argument("--diagnostic-windows", type=int, default=32)
    parser.add_argument("--initial-stream-contrast", type=float, default=0.01,
                        help="Write multipliers [1+3c,1-c,1-c,1-c]; default c=0.01")
    parser.add_argument("--initial-expert-read-contrast", type=float, default=0,
                        help="D-only nonuniform expert read initialization; zero preserves exact lift")
    parser.add_argument("--fp32-read", action="store_true",
                        help="C/D-only FP32 read weights and accumulation under BF16 autocast")
    parser.add_argument("--train-read-maps-only", choices=["bias", "all"],
                        help="Freeze all but C/D read biases or the complete read maps")
    parser.add_argument("--forward-precision", choices=["bf16", "fp32"], default="bf16",
                        help="Use BF16 autocast (default) or FP32 for the entire train/eval forward")
    parser.add_argument("--train-specialists-only", action="store_true",
                        help="Freeze all except routed experts and routers in C")
    parser.add_argument("--save-specialists", type=Path,
                        help="Save only the trained C routed experts and routers")
    parser.add_argument("--load-specialists", type=Path,
                        help="Start C/D from identical pretrained routed experts and routers")
    args = parser.parse_args()
    if args.diagnose and (args.mode != "D" or not 1 <= args.diagnostic_windows <= args.eval_batches):
        raise ValueError("Diagnostics require mode D and 1..eval-batches diagnostic windows")
    if args.initial_stream_contrast != 0.01 and (args.mode in ("dense", "A") or
                                                  not 0 < args.initial_stream_contrast < 1 / 3):
        raise ValueError("Non-default stream contrast requires B/C/D and 0 < c < 1/3")
    if args.initial_expert_read_contrast and (args.mode != "D" or
                                               not 0 < args.initial_expert_read_contrast <= 1):
        raise ValueError("Read contrast requires D and 0 < contrast <= 1")
    if args.fp32_read and args.mode not in ("C", "D"):
        raise ValueError("FP32 read comparison requires C or D")
    if args.train_read_maps_only and (args.mode not in ("C", "D") or not args.fp32_read):
        raise ValueError("Read-only comparison requires C/D and --fp32-read")
    if args.train_specialists_only and (args.mode != "C" or args.train_read_maps_only or
                                        args.load_specialists):
        raise ValueError("Specialist preparation requires C without a read-only/load flag")
    if args.save_specialists and not args.train_specialists_only:
        raise ValueError("Saving specialists requires --train-specialists-only")
    if args.load_specialists and not args.train_read_maps_only:
        raise ValueError("Loading specialists requires read-only C/D training")
    fork_root = os.environ.get("TRANSFORMERS_FORK_ROOT")
    if fork_root and not Path(transformers.__file__).resolve().is_relative_to(Path(fork_root).resolve()):
        raise RuntimeError(f"Transformers imported from {transformers.__file__}, not {fork_root}")
    if (args.steps < 2 or args.eval_batches < 1 or args.eval_every < 1 or
            not 0 <= args.warmup_steps < args.steps):
        raise ValueError("Need at least two steps and positive evaluation settings")
    if not math.isfinite(args.aux_weight) or args.aux_weight < 0:
        raise ValueError("Auxiliary weight must be finite and nonnegative")
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
    if args.mode in "BCD" and args.initial_stream_contrast != 0.01:
        set_initial_stream_contrast(model, args.initial_stream_contrast)
    with torch.no_grad():
        lifted = forward_batch(model, args.mode, probe, probe)[0]
    lift_error = (original - lifted).abs().max().item()
    if lift_error > 1e-3:
        raise RuntimeError(f"Qwen warm-start lift mismatch: {lift_error}")
    initial_logits_delta = 0.0
    if args.initial_expert_read_contrast:
        set_initial_expert_read_contrast(model, args.initial_expert_read_contrast)
        with torch.no_grad():
            perturbed = forward_batch(model, args.mode, probe, probe)[0]
        initial_logits_delta = (original - perturbed).abs().max().item()
    if args.fp32_read:
        for moe in model.moes.values():
            moe.fp32_read = True
    if args.load_specialists:
        checkpoint = torch.load(args.load_specialists, map_location="cpu", weights_only=True)
        load_specialists(model, checkpoint, args.initial_stream_contrast)
    if args.train_read_maps_only:
        freeze_except_read_maps(model, args.train_read_maps_only)
    elif args.train_specialists_only:
        freeze_except_specialists(model)
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
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                            enabled=args.forward_precision == "bf16"):
            losses = [forward_batch(model, args.mode, x, y)[1].item() for x, y in validation]
        model.train()
        return sum(losses) / len(losses)

    initial = evaluate()
    print(json.dumps({"mode": args.mode, "seed": args.seed, "step": 0,
                      "val_nll": initial, "lift_max_abs": lift_error,
                      "initial_stream_contrast": args.initial_stream_contrast,
                      "initial_expert_read_contrast": args.initial_expert_read_contrast,
                      "initial_logits_max_abs_delta": initial_logits_delta,
                      "fp32_read": args.fp32_read,
                      "forward_precision": args.forward_precision,
                      "train_read_maps_only": args.train_read_maps_only,
                      "train_specialists_only": args.train_specialists_only,
                      "aux_weight": args.aux_weight,
                      "load_specialists": str(args.load_specialists) if args.load_specialists else None,
                      "params": sum(p.numel() for p in model.parameters()),
                      "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                      "gpu": torch.cuda.get_device_name(0),
                      "transformers_source": str(Path(transformers.__file__).resolve()),
                      "transformers_version": transformers.__version__}), flush=True)
    if args.diagnose:
        print(json.dumps({"mode": args.mode, "seed": args.seed, "step": 0,
                          "diagnostic": diagnose(model, args.mode,
                                                  validation[:args.diagnostic_windows],
                                                  forward_batch)}), flush=True)
    torch.cuda.reset_peak_memory_stats()
    train_elapsed = 0.0
    start = time.monotonic()
    for step in range(args.steps):
        lr = learning_rate(args.lr, step, args.steps, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = get_batch(train, train_rng, args.batch_size, args.seq_len, "cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=args.forward_precision == "bf16"):
            _, loss, aux = forward_batch(model, args.mode, x, y)
            total = loss + args.aux_weight * aux
        if not torch.isfinite(total):
            raise RuntimeError(f"Non-finite loss at step {step}")
        total.backward()
        if args.diagnose and step in (0, 99, args.steps - 1):
            print(json.dumps({"mode": args.mode, "seed": args.seed, "step": step + 1,
                              "gradient_diagnostic": gradient_probe(model)}), flush=True)
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
                record["mechanism"] = mechanism_metrics(model, args.mode, validation[0][0],
                                                         args.forward_precision)
                if args.diagnose:
                    record["diagnostic"] = diagnose(model, args.mode,
                                                    validation[:args.diagnostic_windows],
                                                    forward_batch)
                    record["diagnostic_fp32"] = diagnose(model, args.mode,
                                                         validation[:min(8, args.diagnostic_windows)],
                                                         forward_batch, precision="fp32")
            print(json.dumps(record), flush=True)
            start = time.monotonic()
    if args.save_specialists:
        selected = specialist_parameters(model)
        args.save_specialists.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"source_mode": "C", "seed": args.seed, "steps": args.steps,
                    "initial_stream_contrast": args.initial_stream_contrast,
                    "parameters": {name: param.detach().cpu() for name, param in selected.items()}},
                   args.save_specialists)


if __name__ == "__main__":
    main()
