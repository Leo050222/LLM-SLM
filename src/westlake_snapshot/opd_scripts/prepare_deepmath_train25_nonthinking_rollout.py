#!/usr/bin/env python3
"""Prepare immutable modulo shards for the DeepMath train-25% rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


EXPECTED_ROWS = 14_208
EXPECTED_PADDING = 50
EXPECTED_SOURCE_SHA256 = (
    "fdcb58732d22f5ab27d111f07c8746836d983c5d15d1d800970154ea1f3cfc91"
)
NUM_SHARDS = 16
ROWS_PER_SHARD = 888


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def flatten(index: int, row: dict[str, Any]) -> dict[str, Any]:
    extra = row["extra_info"]
    reward = row["reward_model"]
    return {
        "dataset_index": index,
        "problem_id": str(extra["problem_id"]),
        "problem": str(extra["question"]),
        "solution": str(reward["ground_truth"]),
        "difficulty": float(extra["difficulty"]),
        "topic": str(extra["topic"]),
        "source_index": int(extra["source_index"]),
        "padded_duplicate": bool(extra["padded_duplicate"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output root: {args.output_root}")
    observed_source_sha = sha256(args.source)
    if observed_source_sha != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"Source SHA mismatch: {observed_source_sha}")
    source_table = pq.read_table(args.source)
    source_rows = source_table.to_pylist()
    if len(source_rows) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} rows, got {len(source_rows)}")
    flat_rows = [flatten(index, row) for index, row in enumerate(source_rows)]
    if sum(row["padded_duplicate"] for row in flat_rows) != EXPECTED_PADDING:
        raise RuntimeError("Unexpected padding count")
    if [row["dataset_index"] for row in flat_rows] != list(range(EXPECTED_ROWS)):
        raise RuntimeError("Dataset indices are not contiguous")

    temporary_root = args.output_root.with_name(args.output_root.name + ".preparing")
    if temporary_root.exists():
        raise FileExistsError(f"Refusing stale temporary root: {temporary_root}")
    inputs = temporary_root / "inputs"
    shards_root = temporary_root / "shards"
    inputs.mkdir(parents=True)
    shards_root.mkdir()
    schema = pa.schema(
        [
            ("dataset_index", pa.int64()),
            ("problem_id", pa.string()),
            ("problem", pa.string()),
            ("solution", pa.string()),
            ("difficulty", pa.float64()),
            ("topic", pa.string()),
            ("source_index", pa.int64()),
            ("padded_duplicate", pa.bool_()),
        ]
    )
    manifest: dict[str, Any] = {
        "status": "DEEPMATH_TRAIN25_NATIVE_NONTHINKING_PREPARED",
        "source": str(args.source),
        "source_sha256": observed_source_sha,
        "rows": EXPECTED_ROWS,
        "padding_rows": EXPECTED_PADDING,
        "num_shards": NUM_SHARDS,
        "rows_per_shard": ROWS_PER_SHARD,
        "sharding": "dataset_index modulo 16",
        "shards": [],
    }
    try:
        for shard_index in range(NUM_SHARDS):
            shard_rows = [
                row for row in flat_rows if row["dataset_index"] % NUM_SHARDS == shard_index
            ]
            if len(shard_rows) != ROWS_PER_SHARD:
                raise RuntimeError(
                    f"Shard {shard_index} has {len(shard_rows)}, expected {ROWS_PER_SHARD}"
                )
            shard_path = inputs / f"shard_{shard_index:02d}.parquet"
            pq.write_table(
                pa.Table.from_pylist(shard_rows, schema=schema),
                shard_path,
                compression="zstd",
            )
            manifest["shards"].append(
                {
                    "shard_index": shard_index,
                    "path": str(
                        args.output_root / "inputs" / f"shard_{shard_index:02d}.parquet"
                    ),
                    "rows": len(shard_rows),
                    "padding_rows": sum(row["padded_duplicate"] for row in shard_rows),
                    "sha256": sha256(shard_path),
                    "min_dataset_index": shard_rows[0]["dataset_index"],
                    "max_dataset_index": shard_rows[-1]["dataset_index"],
                }
            )
        atomic_json(temporary_root / "prepare_manifest.json", manifest)
        temporary_root.replace(args.output_root)
    except BaseException:
        # Preserve evidence only if files were written; never leave the canonical root.
        if temporary_root.exists() and not any(temporary_root.rglob("*")):
            shutil.rmtree(temporary_root)
        raise
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    print("DEEPMATH_TRAIN25_NATIVE_NONTHINKING_PREPARED")


if __name__ == "__main__":
    main()
