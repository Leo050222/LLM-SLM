#!/usr/bin/env python3
"""Create deterministic modulo shards for the fixed DeepMath GE6 test split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    args = parser.parse_args()

    actual_sha = sha256(args.input)
    if actual_sha != args.expected_sha256:
        raise RuntimeError(f"Input SHA mismatch: {actual_sha}")
    source_rows = pq.read_table(args.input).to_pylist()
    if len(source_rows) != 1000:
        raise RuntimeError(f"Expected 1000 rows, got {len(source_rows)}")
    flat_rows = []
    for dataset_index, row in enumerate(source_rows):
        extra = row["extra_info"]
        reward = row["reward_model"]
        flat_rows.append(
            {
                "dataset_index": dataset_index,
                "id": str(extra["problem_id"]),
                "problem": str(extra["question"]),
                "solution": str(reward["ground_truth"]),
                "difficulty": float(extra["difficulty"]),
                "topic": str(extra["topic"]),
                "source_index": int(extra["source_index"]),
            }
        )

    import pandas as pd

    frame = pd.DataFrame(flat_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    flat_output = args.output_dir / "test_1000_flat.parquet"
    frame.to_parquet(flat_output, index=False, compression="zstd")
    shard_rows: list[int] = []
    shard_sha256: list[str] = []
    for shard_index in range(args.num_shards):
        shard = frame[frame["dataset_index"].astype(int) % args.num_shards == shard_index]
        shard = shard.sort_values("dataset_index").reset_index(drop=True)
        output = args.output_dir / f"test_1000.shard{shard_index}.parquet"
        shard.to_parquet(output, index=False, compression="zstd")
        shard_rows.append(len(shard))
        shard_sha256.append(sha256(output))

    if sum(shard_rows) != 1000 or shard_rows != [250] * args.num_shards:
        raise RuntimeError(f"Unexpected shard rows: {shard_rows}")
    manifest = {
        "status": "DEEPMATH_GE6_EVAL_SHARDS_OK",
        "input": str(args.input),
        "input_sha256": actual_sha,
        "num_rows": 1000,
        "flat_file": str(flat_output),
        "flat_sha256": sha256(flat_output),
        "num_shards": args.num_shards,
        "shard_rows": shard_rows,
        "shard_sha256": shard_sha256,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
