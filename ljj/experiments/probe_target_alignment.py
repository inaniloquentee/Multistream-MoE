"""Train-only read-direction gradients tested on held-out Qwen windows."""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import Qwen3ForCausalLM

from experiments.qwen_diagnostics import set_initial_stream_contrast
from experiments.train_qwen import forward_batch, get_batch
from multistream_moe.qwen_reference import QwenMultiStream


def project_independent_read_bias(gradient):
    """Remove tied-expert and softmax-null components of a (5, 4) bias gradient."""
    if gradient.shape != (5, 4):
        raise ValueError("Expected one shared plus four routed, four-stream read biases")
    stream = gradient.new_tensor([3, -1, -1, -1])
    projected = ((gradient @ stream) / (stream @ stream))[:, None] * stream[None, :]
    return projected - projected.mean(dim=0, keepdim=True)


def downhill_direction(gradients):
    maximum = max(t.abs().max().item() for t in gradients.values())
    if not math.isfinite(maximum) or maximum <= 1e-12:
        raise ValueError("No finite independent read gradient")
    return {name: -gradient / maximum for name, gradient in gradients.items()}


def dot(left, right):
    return sum((left[name] * right[name]).sum().item() for name in left)


def gradients(model, batches):
    model.zero_grad(set_to_none=True)
    for x, y in batches:
        loss = forward_batch(model, "D", x, y)[1]
        (loss / len(batches)).backward()
    result = {name: project_independent_read_bias(moe.read_bias.grad.detach()).clone()
              for name, moe in model.moes.items()}
    model.zero_grad(set_to_none=True)
    return result


def losses(model, batches, precision):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                         enabled=precision == "bf16"):
        return [forward_batch(model, "D", x, y)[1].item() for x, y in batches]


def paired(reference, changed):
    differences = [b - a for a, b in zip(reference, changed)]
    mean = sum(differences) / len(differences)
    se = math.sqrt(sum((d - mean) ** 2 for d in differences) /
                   (len(differences) * (len(differences) - 1)))
    return {"delta_nll": mean, "window_se": se}


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
                      "train_gradient_windows": 32, "validation_gradient_windows": 32,
                      "validation_windows": 128,
                      "direction_seeds": [42, 43, 44],
                      "amplitudes": [-0.05, 0.01, 0.05, 0.25]}), flush=True)
    train_gradients = {}
    for seed in (42, 43, 44):
        rng = np.random.default_rng(35000 + seed)
        batches = [get_batch(train, rng, 2, 128, "cuda") for _ in range(32)]
        train_gradients[seed] = gradients(model, batches)
    val_gradient = gradients(model, validation[:32])
    baselines = {precision: losses(model, validation, precision)
                 for precision in ("fp32", "bf16")}
    original = {name: moe.read_bias.detach().clone() for name, moe in model.moes.items()}
    for seed, train_gradient in train_gradients.items():
        direction = downhill_direction(train_gradient)
        train_norm = math.sqrt(dot(train_gradient, train_gradient))
        val_norm = math.sqrt(dot(val_gradient, val_gradient))
        print(json.dumps({"seed": seed, "train_projected_gradient_norm": train_norm,
                          "val_projected_gradient_norm": val_norm,
                          "train_val_projected_gradient_cosine":
                          dot(train_gradient, val_gradient) / (train_norm * val_norm),
                          "train_directional_derivative": dot(train_gradient, direction),
                          "val_directional_derivative": dot(val_gradient, direction)}), flush=True)
        for amplitude in (-0.05, 0.01, 0.05, 0.25):
            try:
                with torch.no_grad():
                    for name, moe in model.moes.items():
                        moe.read_bias.copy_(original[name] + amplitude * direction[name])
                fp32 = losses(model, validation, "fp32")
                bf16 = losses(model, validation, "bf16")
            finally:
                with torch.no_grad():
                    for name, moe in model.moes.items():
                        moe.read_bias.copy_(original[name])
            for precision, windows, trial in (("fp32", 32, fp32[:32]),
                                              ("fp32", 128, fp32),
                                              ("bf16", 32, bf16[:32]),
                                              ("bf16", 128, bf16)):
                reference = baselines[precision][:windows]
                print(json.dumps({"seed": seed, "amplitude": amplitude,
                                  "precision": precision, "windows": windows,
                                  "baseline_nll": sum(reference) / windows,
                                  "trial_nll": sum(trial) / windows,
                                  "paired": paired(reference, trial)}), flush=True)


if __name__ == "__main__":
    main()
