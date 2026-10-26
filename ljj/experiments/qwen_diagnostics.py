"""Paired counterfactuals and activation probes for the Qwen multi-stream read path."""

from contextlib import contextmanager
import math

import torch
from torch.nn import functional as F


def _rms(x):
    return x.float().square().mean().sqrt().item()


def set_initial_stream_contrast(model, contrast):
    """Change stream spread without changing the mean write multiplier of one."""
    if model.mode == "A" or not 0 < contrast < 1 / 3:
        raise ValueError("Require a multi-stream model and 0 < contrast < 1/3")
    if model.streams != 4:
        raise ValueError("This warm-start intervention expects four streams")
    with torch.no_grad():
        for moe in model.moes.values():
            values = moe.write_bias.new_tensor([1 + 3 * contrast, *([1 - contrast] * 3)])
            moe.write_bias.copy_(torch.logit(values / 2).expand_as(moe.write_bias))


@contextmanager
def replace_maps(model, kind, replacement):
    """Temporarily ablate D's read maps or C/D's write maps in place."""
    if kind not in ("read", "write") or replacement not in ("tied", "uniform"):
        raise ValueError((kind, replacement))
    if kind == "write" and replacement == "uniform":
        raise ValueError("Uniform write is not a valid isolated read intervention")
    snapshots = []
    with torch.no_grad():
        for moe in model.moes.values():
            weight = getattr(moe, f"{kind}_weight")
            bias = getattr(moe, f"{kind}_bias")
            snapshots.append((weight, bias, weight.detach().clone(), bias.detach().clone()))
            if replacement == "tied":
                weight.copy_(weight.mean(dim=0, keepdim=True).expand_as(weight))
                bias.copy_(bias.mean(dim=0, keepdim=True).expand_as(bias))
            elif kind == "read":
                weight.zero_()
                bias.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            for weight, bias, saved_weight, saved_bias in snapshots:
                weight.copy_(saved_weight)
                bias.copy_(saved_bias)


def _losses(model, validation, forward_batch, mode, precision):
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16")
    with torch.no_grad(), autocast:
        return [forward_batch(model, mode, x, y)[1].item() for x, y in validation]


def _paired(baseline, changed):
    differences = [b - a for a, b in zip(baseline, changed)]
    mean = sum(differences) / len(differences)
    se = (math.sqrt(sum((d - mean) ** 2 for d in differences) /
                    (len(differences) - 1) / len(differences))
          if len(differences) > 1 else None)
    return {"delta_nll": mean, "window_se": se,
            "ci95_window": [mean - 1.96 * se, mean + 1.96 * se] if se is not None else None}


def activation_probe(model, tokens, precision):
    """Measure effective expert reads at each MoE input, not just parameter norms."""
    layers = {}
    handles = []

    def capture(name, moe):
        def hook(_module, args):
            x = args[0]
            flat = x.reshape(-1, moe.streams, x.shape[-1])
            context = moe.context_norm(flat.mean(dim=1))
            reads = torch.stack([moe.read(flat, context, j)
                                 for j in range(moe.n_experts + 1)], dim=1)
            shared = reads[:, :1, :]
            weights = torch.stack([
                F.linear(context, moe.read_weight[j], moe.read_bias[j]).softmax(-1)
                for j in range(moe.n_experts + 1)
            ], dim=1)
            with torch.autocast(device_type=x.device.type, enabled=False):
                fp32_weights = torch.stack([
                    F.linear(context.float(), moe.read_weight[j].float(),
                             moe.read_bias[j].float()).softmax(-1)
                    for j in range(moe.n_experts + 1)
                ], dim=1)
                centered = flat.float() - flat.float().mean(dim=1, keepdim=True)
                read_signal = torch.einsum("nsd,njs->njd", centered,
                                           fp32_weights[:, 1:] - fp32_weights[:, :1])
            indices = moe.router(context).topk(moe.top_k, dim=-1).indices
            loads = torch.bincount(indices.flatten(), minlength=moe.n_experts)
            difference = reads[:, 1:, :] - shared
            layers[name] = {
                "input_centered_rms": _rms(flat - flat.mean(dim=1, keepdim=True)),
                "input_mean_rms": _rms(flat.mean(dim=1)),
                "read_expert_delta_rms": _rms(difference),
                "read_signal_fp32_same_input_rms": _rms(read_signal),
                "read_expert_exact_equal_fraction": (difference == 0).float().mean().item(),
                "read_weight_expert_delta_rms": _rms(weights[:, 1:] - weights[:, :1]),
                "read_weight_uniform_delta_rms": _rms(weights - 1 / moe.streams),
                "route_load_fraction": (loads.float() / indices.numel()).tolist(),
            }
        return hook

    try:
        for name, moe in model.moes.items():
            handles.append(moe.register_forward_pre_hook(capture(name, moe)))
        autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16")
        with torch.no_grad(), autocast:
            model(tokens)
    finally:
        for handle in handles:
            handle.remove()
    return layers


def diagnose(model, mode, validation, forward_batch, precision="bf16"):
    if mode != "D":
        raise ValueError("The read-map counterfactual requires mode D")
    was_training = model.training
    model.eval()
    try:
        baseline = _losses(model, validation, forward_batch, mode, precision)
        result = {"precision": precision, "eval_windows": len(validation),
                  "baseline_nll": sum(baseline) / len(baseline),
                  "activation": activation_probe(model, validation[0][0], precision)}
        for kind, replacement in (("read", "tied"), ("read", "uniform"),
                                  ("write", "tied")):
            with replace_maps(model, kind, replacement):
                changed = _losses(model, validation, forward_batch, mode, precision)
            result[f"{kind}_{replacement}"] = _paired(baseline, changed)
        return result
    finally:
        model.train(was_training)


def gradient_probe(model):
    if not hasattr(model, "moes"):
        return {}
    return {name: {"read_grad_rms": _rms(moe.read_weight.grad)
                    if moe.read_weight.grad is not None else None,
                    "write_grad_rms": _rms(moe.write_weight.grad)
                    if moe.write_weight.grad is not None else None}
            for name, moe in model.moes.items()}
