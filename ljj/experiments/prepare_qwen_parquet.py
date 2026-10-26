"""Build a reproducible document split from a pinned FineWeb-Edu Parquet shard."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from experiments.prepare_qwen import DATASET, MODEL, MODEL_REVISION, sha256


DATASET_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
SHARD = "sample/10BT/000_00000.parquet"
ROOT = Path(__file__).resolve().parent.parent


def split_is_val(doc_id):
    digest = hashlib.sha256(doc_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 10 == 0


def tokenize_shard(parquet_path, tokenizer, output, train_target, val_target):
    import pyarrow.parquet as pq

    output.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0}
    documents = {"train": 0, "val": 0}
    skipped = {"empty": 0, "duplicate_id": 0, "duplicate_text": 0}
    seen_ids, seen_text = set(), set()
    rows_scanned = 0
    names = ("train.bin", "val.bin", "train.ids", "val.ids")
    streams = {name: (output / (name + ".partial")).open("wb") for name in names}
    try:
        parquet = pq.ParquetFile(parquet_path)
        if not {"id", "text"}.issubset(parquet.schema_arrow.names):
            raise ValueError("Parquet shard must contain id and text")
        finished = False
        for batch in parquet.iter_batches(batch_size=128, columns=["id", "text"]):
            columns = batch.to_pydict()
            for raw_id, text in zip(columns["id"], columns["text"]):
                rows_scanned += 1
                if raw_id is None or not text:
                    skipped["empty"] += 1
                    continue
                doc_id = str(raw_id)
                if doc_id in seen_ids:
                    skipped["duplicate_id"] += 1
                    continue
                seen_ids.add(doc_id)
                text_hash = hashlib.sha256(text.encode("utf-8")).digest()
                if text_hash in seen_text:
                    skipped["duplicate_text"] += 1
                    continue
                seen_text.add(text_hash)
                split = "val" if split_is_val(doc_id) else "train"
                tokens = tokenizer.encode(text, add_special_tokens=False)
                tokens.append(tokenizer.eos_token_id)
                if not all(0 <= token < 2**32 for token in tokens):
                    raise ValueError("Token ID does not fit uint32")
                np.asarray(tokens, dtype=np.uint32).tofile(streams[split + ".bin"])
                streams[split + ".ids"].write((doc_id + "\n").encode("utf-8"))
                counts[split] += len(tokens)
                documents[split] += 1
                if rows_scanned % 5000 == 0:
                    print(json.dumps({"rows_scanned": rows_scanned, "tokens": counts}), flush=True)
                if counts["train"] >= train_target and counts["val"] >= val_target:
                    finished = True
                    break
            if finished:
                break
    finally:
        for stream in streams.values():
            stream.close()
    if counts["train"] < train_target or counts["val"] < val_target:
        raise RuntimeError(f"Shard too small for target token counts: {counts}")
    for name in names:
        os.replace(output / (name + ".partial"), output / name)
    result = {"rows_scanned": rows_scanned, "train_tokens": counts["train"],
              "val_tokens": counts["val"], "train_documents": documents["train"],
              "val_documents": documents["val"], "skipped": skipped}
    for name in names:
        result[name + "_sha256"] = sha256(output / name)
    return result


def verify_output(output, source=None, tokenizer_file=None):
    manifest = json.loads((output / "manifest.json").read_text())
    if (manifest["dataset_revision"] != DATASET_REVISION or
            manifest["model_revision"] != MODEL_REVISION or
            manifest["shard"] != SHARD):
        raise ValueError("Unexpected dataset, model or shard revision")
    source = source or ROOT / "cache/fineweb-edu-pinned" / SHARD
    if sha256(source) != manifest["shard_sha256"]:
        raise ValueError("Parquet SHA256 mismatch")
    tokenizer_file = tokenizer_file or ROOT / "cache/qwen3-0.6b-base/tokenizer.json"
    if sha256(tokenizer_file) != manifest["tokenizer_json_sha256"]:
        raise ValueError("Tokenizer SHA256 mismatch")
    ids = {}
    for split in ("train", "val"):
        for name in (split + ".bin", split + ".ids"):
            if sha256(output / name) != manifest[name + "_sha256"]:
                raise ValueError(f"SHA256 mismatch: {name}")
        ids[split] = (output / (split + ".ids")).read_text().splitlines()
        if (len(ids[split]) != manifest[split + "_documents"] or
                len(set(ids[split])) != len(ids[split]) or
                (output / (split + ".bin")).stat().st_size != 4 * manifest[split + "_tokens"]):
            raise ValueError(f"Invalid {split} document or token counts")
    if set(ids["train"]) & set(ids["val"]):
        raise ValueError("Document overlap between train and validation")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/qwen-fineweb-pinned")
    parser.add_argument("--train-tokens", type=int, default=20_000_000)
    parser.add_argument("--val-tokens", type=int, default=2_000_000)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or min(args.train_tokens, args.val_tokens) < 1:
        raise ValueError("Output must be within ljj/ and token targets must be positive")
    if args.verify_only:
        print(json.dumps(verify_output(output), sort_keys=True), flush=True)
        return
    sys.path.insert(0, str(ROOT / "_deps"))
    os.environ.setdefault("HF_HOME", str(ROOT / "cache"))
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    model_dir = ROOT / "cache/qwen3-0.6b-base"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    parquet_path = Path(hf_hub_download(
        repo_id=DATASET, repo_type="dataset", revision=DATASET_REVISION,
        filename=SHARD, local_dir=str(ROOT / "cache/fineweb-edu-pinned")))
    source_sha = sha256(parquet_path)
    result = tokenize_shard(parquet_path, tokenizer, output,
                            args.train_tokens, args.val_tokens)
    manifest = {"dataset": DATASET, "dataset_revision": DATASET_REVISION,
                "shard": SHARD, "shard_sha256": source_sha,
                "model": MODEL, "model_revision": MODEL_REVISION,
                "tokenizer_json_sha256": sha256(model_dir / "tokenizer.json"),
                "deduplication": "exact document ID and UTF-8 text SHA256, first occurrence",
                "split": "first 64 bits of SHA256(document ID) modulo 10 == 0 for validation",
                "dtype": "uint32", **result}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(verify_output(output, parquet_path), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
