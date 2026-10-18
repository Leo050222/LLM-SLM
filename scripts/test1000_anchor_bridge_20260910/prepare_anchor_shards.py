#!/usr/bin/env python3
"""Strictly shard the frozen test1000 canonical Student rollout by dataset index."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-rows", type=int, default=1000)
    parser.add_argument("--num-shards", type=int, default=8)
    args = parser.parse_args()

    observed_sha = sha256(args.source)
    if observed_sha != args.expected_source_sha256:
        raise ValueError(f"source SHA mismatch: {observed_sha}")
    with args.source.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} rows, got {len(rows)}")
    indices = [int(row["dataset_index"]) for row in rows]
    if sorted(indices) != list(range(args.expected_rows)):
        raise ValueError("dataset_index coverage is not exactly 0..expected_rows-1")
    required = {"problem_id", "solution", "prompt_token_ids", "response_token_ids"}
    for row in rows:
        missing = required - row.keys()
        if missing or not row["prompt_token_ids"] or not row["response_token_ids"]:
            raise ValueError(f"invalid row {row.get('dataset_index')}: missing={sorted(missing)}")

    shard_manifest = []
    for shard_id in range(args.num_shards):
        shard_rows = [row for row in rows if int(row["dataset_index"]) % args.num_shards == shard_id]
        path = args.output_root / "shards" / f"shard_{shard_id:02d}" / "rollouts.jsonl"
        atomic_jsonl(path, shard_rows)
        shard_manifest.append({"shard_id": shard_id, "rows": len(shard_rows), "sha256": sha256(path)})

    manifest = {
        "status": "TEST1000_ANCHOR_SOURCE_SHARDS_COMPLETE",
        "source": str(args.source),
        "source_sha256": observed_sha,
        "expected_rows": args.expected_rows,
        "num_shards": args.num_shards,
        "shards": shard_manifest,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = args.output_root / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output_root / "manifest.json")
    (args.output_root / "SOURCE_SHARDS_COMPLETE").touch()
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
