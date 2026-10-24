"""Pin Qwen weights/tokenizer and build an explicitly non-publication FineWeb proxy."""

import argparse
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np


MODEL = "Qwen/Qwen3-0.6B-Base"
MODEL_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
DATASET = "HuggingFaceFW/fineweb-edu"


def request_rows(offset, length):
    query = urllib.parse.urlencode({"dataset": DATASET, "config": "sample-10BT",
                                    "split": "train", "offset": offset, "length": length})
    url = "https://datasets-server.huggingface.co/rows?" + query
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                return json.load(response)["rows"]
        except (OSError, ValueError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--output", type=Path, default=root / "data/qwen-fineweb-proxy")
    parser.add_argument("--max-rows", type=int, default=3000)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(root / "cache"))
    from huggingface_hub import HfApi, snapshot_download
    import transformers
    from transformers import AutoTokenizer

    fork_root = os.environ.get("TRANSFORMERS_FORK_ROOT")
    if fork_root and not Path(transformers.__file__).resolve().is_relative_to(Path(fork_root).resolve()):
        raise RuntimeError(f"Transformers imported from {transformers.__file__}, not {fork_root}")

    model_dir = root / "cache/qwen3-0.6b-base"
    snapshot_download(repo_id=MODEL, revision=MODEL_REVISION, local_dir=str(model_dir),
                      allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model",
                                      "vocab*", "merges*"])
    print(json.dumps({"model": MODEL, "revision": MODEL_REVISION,
                      "model_dir": str(model_dir)}), flush=True)
    if args.download_only:
        return
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    dataset_revision = HfApi().dataset_info(DATASET).sha
    train_ids, val_ids = [], []
    train_docs, val_docs, skipped = [], [], 0
    for offset in range(0, args.max_rows, 100):
        rows = request_rows(offset, min(100, args.max_rows - offset))
        if not rows:
            break
        for item in rows:
            doc = item["row"]
            if not doc.get("text") or item.get("truncated_cells"):
                skipped += 1
                continue
            doc_id = str(doc["id"])
            digest = hashlib.sha256(doc_id.encode()).digest()
            is_val = int.from_bytes(digest[:8], "big") % 10 == 0
            tokens = tokenizer.encode(doc["text"], add_special_tokens=False)
            tokens.append(tokenizer.eos_token_id)
            (val_ids if is_val else train_ids).extend(tokens)
            (val_docs if is_val else train_docs).append(doc_id)
        print(json.dumps({"rows_seen": offset + len(rows), "train_tokens": len(train_ids),
                          "val_tokens": len(val_ids), "skipped": skipped}), flush=True)
    if min(len(train_ids), len(val_ids)) < 8192:
        raise RuntimeError("Insufficient data; increase --max-rows")
    args.output.mkdir(parents=True, exist_ok=True)
    for split, tokens in (("train", train_ids), ("val", val_ids)):
        np.asarray(tokens, dtype=np.uint32).tofile(args.output / f"{split}.bin")
    manifest = {"purpose": "non-publication proxy; dataset viewer may truncate and drift",
                "model": MODEL, "model_revision": MODEL_REVISION,
                "dataset": DATASET, "dataset_revision_observed": dataset_revision,
                "source": "dataset viewer rows, offsets [0,max_rows); not a pinned parquet snapshot",
                "max_rows": args.max_rows, "skipped": skipped,
                "vocab_size": tokenizer.vocab_size, "dtype": "uint32",
                "train_documents": len(train_docs), "val_documents": len(val_docs),
                "train_tokens": len(train_ids), "val_tokens": len(val_ids)}
    for name, ids in (("train", train_docs), ("val", val_docs)):
        manifest[f"{name}_ids_sha256"] = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        manifest[f"{name}_bin_sha256"] = sha256(args.output / f"{name}.bin")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
