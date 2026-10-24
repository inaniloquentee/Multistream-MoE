"""CPU checkpoint-equivalence check when H20 allocation is unavailable."""

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import Qwen3ForCausalLM

from qwen_reference import QwenMultiStream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--modes", default="ABCD")
    args = parser.parse_args()
    fork_root = os.environ.get("TRANSFORMERS_FORK_ROOT")
    if fork_root and not Path(transformers.__file__).resolve().is_relative_to(Path(fork_root).resolve()):
        raise RuntimeError(f"Transformers imported from {transformers.__file__}, not {fork_root}")
    print(json.dumps({"transformers_source": str(Path(transformers.__file__).resolve()),
                      "transformers_version": transformers.__version__}), flush=True)
    torch.set_num_threads(4)
    raw = np.memmap(args.data / "val.bin", dtype=np.uint32, mode="r")
    tokens = torch.tensor(raw[:8].astype(np.int64))[None, :]
    for mode in args.modes:
        base = Qwen3ForCausalLM.from_pretrained(str(args.model), dtype=torch.float32,
                                                attn_implementation="sdpa").eval()
        with torch.no_grad():
            expected = base(tokens).logits
        torch.manual_seed(42)
        model = QwenMultiStream(base, mode, n_experts=4).eval()
        with torch.no_grad():
            logits, _, _, streams = model(tokens)
        error = (logits - expected).abs().max().item()
        if error > 1e-3:
            raise RuntimeError(f"{mode}: lifted Qwen differs from checkpoint: {error}")
        mean = streams.mean(dim=2)
        centered = streams - mean.unsqueeze(2)
        print(json.dumps({"mode": mode, "max_logit_abs_error": error,
                          "stream_difference": (streams[:, :, 0] - streams[:, :, -1]).abs().max().item(),
                          "stream_rms": centered.square().mean().sqrt().item(),
                          "mean_rms": mean.square().mean().sqrt().item(),
                          "params": sum(p.numel() for p in model.parameters())}), flush=True)
        del logits, expected, streams, model, base
        gc.collect()


if __name__ == "__main__":
    main()
