"""Small Parquet fixture for the pinned, document-disjoint tokenizer path."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.prepare_qwen import sha256
from experiments.prepare_qwen_parquet import (DATASET_REVISION, MODEL_REVISION,
                                               ROOT, SHARD, split_is_val,
                                               tokenize_shard, verify_output)


class Tokenizer:
    eos_token_id = 0

    @staticmethod
    def encode(text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(char) for char in text]


class ParquetPreparationTests(unittest.TestCase):
    def test_split_deduplication_and_repeatability(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        train = [str(i) for i in range(100) if not split_is_val(str(i))]
        val = next(str(i) for i in range(100) if split_is_val(str(i)))
        rows = {"id": [train[0], train[0], train[1], None, val, train[2]],
                "text": ["first", "same id", "first", "empty", "validation", "second"]}
        cache = ROOT / "cache/tests"
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as directory:
            root = Path(directory)
            shard = root / "input.parquet"
            pq.write_table(pa.table(rows), shard)
            output = root / "tokens"
            first = tokenize_shard(shard, Tokenizer(), output, 12, 6)
            self.assertEqual(first["skipped"], {
                "empty": 1, "duplicate_id": 1, "duplicate_text": 1})
            self.assertEqual((first["train_documents"], first["val_documents"]), (2, 1))
            self.assertEqual((output / "train.ids").read_text().splitlines(),
                             [train[0], train[2]])
            self.assertEqual((output / "val.ids").read_text().splitlines(), [val])
            self.assertEqual(len(np.fromfile(output / "train.bin", dtype=np.uint32)),
                             first["train_tokens"])
            self.assertEqual(first, tokenize_shard(shard, Tokenizer(), output, 12, 6))
            tokenizer_file = root / "tokenizer.json"
            tokenizer_file.write_text("fixture")
            manifest = {"dataset_revision": DATASET_REVISION,
                        "model_revision": MODEL_REVISION, "shard": SHARD,
                        "shard_sha256": sha256(shard),
                        "tokenizer_json_sha256": sha256(tokenizer_file), **first}
            (output / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(verify_output(output, shard, tokenizer_file), manifest)
            with (output / "train.bin").open("ab") as stream:
                stream.write(b"0000")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_output(output, shard, tokenizer_file)


if __name__ == "__main__":
    unittest.main()
