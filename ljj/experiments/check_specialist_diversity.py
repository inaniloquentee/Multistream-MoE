"""Inspect routed expert parameter separation in a saved C specialist state."""

import argparse
import json
from pathlib import Path

import torch


def diversity(parameters, layer, n_experts=4):
    weights = torch.stack([parameters[f"moes.{layer}.experts.{j}.mlp.down_proj.weight"]
                           for j in range(1, n_experts + 1)]).float()
    centered = weights - weights.mean(dim=0, keepdim=True)
    return {"routed_down_proj_rms": weights.square().mean().sqrt().item(),
            "routed_expert_centered_rms": centered.square().mean().sqrt().item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    parameters = checkpoint["parameters"]
    print(json.dumps({"checkpoint": str(args.checkpoint), "seed": checkpoint["seed"],
                      "steps": checkpoint["steps"],
                      "layers": {layer: diversity(parameters, layer) for layer in ("1", "3")}}))


if __name__ == "__main__":
    main()
