"""Locate the first mismatch between the expressivity example and Qwen lift."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import Qwen3ForCausalLM

from experiments.qwen_diagnostics import set_initial_stream_contrast
from experiments.train_qwen import get_batch
from multistream_moe.qwen_reference import QwenMultiStream


def rms(x):
    return x.float().square().mean().sqrt().item()


def stream_geometry(x):
    centered = x - x.mean(dim=2, keepdim=True)
    direction = x.new_tensor([3.0, -1.0, -1.0, -1.0]) / 12 ** 0.5
    projected = (centered * direction[None, None, :, None]).sum(dim=2, keepdim=True)
    orthogonal = centered - projected * direction[None, None, :, None]
    return {"centered_rms": rms(centered), "off_axis_rms": rms(orthogonal),
            "last_three_max_abs_difference": max(
                (x[:, :, 1] - x[:, :, j]).abs().max().item() for j in (2, 3))}


def probe(model, tokens):
    captured = {}
    handles = []
    for name, moe in model.moes.items():
        def before(_module, args, layer=name):
            captured[layer] = {"entry": args[0]}

        def after(_module, _args, output, layer=name):
            captured[layer]["exit"] = output[0]

        handles.append(moe.register_forward_pre_hook(before))
        handles.append(moe.register_forward_hook(after))
    try:
        with torch.no_grad():
            model(tokens)
    finally:
        for handle in handles:
            handle.remove()

    layers = {name: {"entry": stream_geometry(item["entry"]),
                     "exit": stream_geometry(item["exit"])}
              for name, item in captured.items()}
    first, second = model.moes["1"], model.moes["3"]
    incoming, outgoing = captured["1"]["entry"], captured["1"]["exit"]
    flat = incoming.reshape(-1, first.streams, incoming.shape[-1])
    context = first.context_norm(flat.mean(dim=1))
    with torch.no_grad():
        shared_input = first.read(flat, context, 0)
        shared_output = first.experts[0](shared_input)
        copy_error = max((first.experts[j](shared_input[:8]) - shared_output[:8])
                         .abs().max().item() for j in range(1, first.n_experts + 1))
        write = first.write(context, 0)
        predicted = flat + (2 * shared_output[:, None, :] * write[:, :, None])
        first_identity_error = (outgoing.reshape_as(predicted) - predicted).abs().max().item()

        second_input = captured["3"]["entry"]
        states = second_input.reshape(-1, second.streams, second_input.shape[-1])
        second_context = second.context_norm(states.mean(dim=1))
        ordinary = second.read(states, second_context, 0)
        uniform_delta = max((second.read(states, second_context, j) - ordinary)
                            .abs().max().item() for j in range(1, second.n_experts + 1))

        # The theoretical counterexample chooses distinct, nonuniform expert reads.
        saved = second.read_bias.detach().clone()
        try:
            second.read_bias[0].copy_(states.new_tensor([0.7, 0.1, 0.1, 0.1]).log())
            second.read_bias[1].copy_(states.new_tensor([0.1, 0.3, 0.3, 0.3]).log())
            left = second.read(states, second_context, 0)
            right = second.read(states, second_context, 1)
            artificial_delta = rms(left - right)
            nonlinear_delta = rms(second.experts[0](left) - second.experts[1](right))
        finally:
            second.read_bias.copy_(saved)

    if (layers["1"]["entry"]["centered_rms"] > 1e-6 or
            first_identity_error > 1e-3 or uniform_delta != 0 or artificial_delta <= 0):
        raise RuntimeError("The expected warm-start algebra did not hold")
    return {"layers": layers,
            "first_moe_copied_expert_max_abs_error": copy_error,
            "first_moe_dense_update_identity_max_abs_error": first_identity_error,
            "second_moe_expert_read_max_abs_difference_at_init": uniform_delta,
            "second_moe_counterexample_read_rms": artificial_delta,
            "second_moe_counterexample_expert_output_rms": nonlinear_delta}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    args = parser.parse_args()
    fork_root = os.environ.get("TRANSFORMERS_FORK_ROOT")
    if fork_root and not Path(transformers.__file__).resolve().is_relative_to(Path(fork_root).resolve()):
        raise RuntimeError(f"Transformers imported from {transformers.__file__}, not {fork_root}")
    if not torch.cuda.is_available() or "H20" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Run the checkpoint probe on an allocated H20")
    raw = np.memmap(args.data / "val.bin", mode="r", dtype=np.uint32)
    tokens, _ = get_batch(raw, np.random.default_rng(2027), 2, args.seq_len, "cuda")
    base = Qwen3ForCausalLM.from_pretrained(str(args.model), dtype=torch.float32,
                                            attn_implementation="sdpa").eval()
    torch.manual_seed(42)
    model = QwenMultiStream(base, "D", n_experts=4).cuda().eval()
    for contrast in (0.01, 0.2):
        set_initial_stream_contrast(model, contrast)
        result = probe(model, tokens)
        print(json.dumps({"initial_stream_contrast": contrast, "seq_len": args.seq_len,
                          "batch_size": 2, "gpu": torch.cuda.get_device_name(0),
                          "transformers_source": str(Path(transformers.__file__).resolve()),
                          "probe": result}), flush=True)


if __name__ == "__main__":
    main()
