#!/usr/bin/env python3
"""Create a deterministic proportional sample stratified by difficulty x topic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REQUIRED_COLUMNS = ("question", "final_answer", "difficulty", "topic")


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def allocate_largest_remainder(counts: pd.Series, sample_size: int) -> pd.Series:
    """Allocate an exact proportional sample across all observed strata."""
    ideal = counts.astype(float) * (sample_size / int(counts.sum()))
    allocation = np.floor(ideal).astype(int)
    allocation = pd.Series(allocation, index=counts.index, dtype="int64")
    remaining = sample_size - int(allocation.sum())
    if remaining:
        remainder = (ideal - allocation).sort_values(ascending=False, kind="stable")
        for stratum in remainder.index[:remaining]:
            allocation.loc[stratum] += 1
    if (allocation > counts).any():
        raise RuntimeError("A stratum was allocated more rows than it contains")
    if int(allocation.sum()) != sample_size:
        raise RuntimeError("Largest-remainder allocation did not reach sample_size")
    return allocation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()

    shards = sorted((args.source_dir / "data").glob("train-*.parquet"))
    if len(shards) != 10:
        raise RuntimeError(f"Expected 10 source shards, found {len(shards)}")
    frames = [pq.read_table(path, columns=list(REQUIRED_COLUMNS)).to_pandas() for path in shards]
    frame = pd.concat(frames, ignore_index=True)
    if len(frame) != 103_022:
        raise RuntimeError(f"Expected 103022 source rows, found {len(frame)}")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise RuntimeError("Required DeepMath fields contain null values")

    frame.insert(0, "source_index", np.arange(len(frame), dtype=np.int64))
    frame["difficulty"] = frame["difficulty"].astype(float)
    frame["topic"] = frame["topic"].astype(str)
    counts = frame.groupby(["difficulty", "topic"], sort=True).size()
    allocation = allocate_largest_remainder(counts, args.sample_size)

    sampled_parts: list[pd.DataFrame] = []
    for (difficulty, topic), take in allocation.items():
        if not take:
            continue
        group = frame[(frame["difficulty"] == difficulty) & (frame["topic"] == topic)]
        # Derive an independent deterministic RNG stream for every stratum.
        digest = hashlib.sha256(
            f"{args.seed}|{difficulty!r}|{topic}".encode("utf-8")
        ).digest()
        group_seed = int.from_bytes(digest[:8], "big", signed=False)
        rng = np.random.default_rng(group_seed)
        positions = np.sort(rng.choice(len(group), size=int(take), replace=False))
        sampled_parts.append(group.iloc[positions])

    sample = pd.concat(sampled_parts, ignore_index=True)
    sample = sample.sort_values("source_index", kind="stable").reset_index(drop=True)
    sample.insert(0, "dataset_index", np.arange(len(sample), dtype=np.int64))
    sample.insert(1, "id", sample["source_index"].map(lambda value: f"deepmath-{value}"))
    sample["problem"] = sample["question"]
    sample["solution"] = sample["final_answer"]
    if len(sample) != args.sample_size or sample["source_index"].duplicated().any():
        raise RuntimeError("Sample size or uniqueness check failed")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    sample.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(args.output)
    shard_paths: list[str] = []
    for shard_index in range(4):
        shard = sample[sample["dataset_index"] % 4 == shard_index].copy()
        shard_path = args.output.with_name(
            f"{args.output.stem}.shard{shard_index}{args.output.suffix}"
        )
        shard.to_parquet(shard_path, index=False, compression="zstd")
        shard_paths.append(str(shard_path))
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()

    sampled_counts = (
        sample.groupby(["difficulty", "topic"], sort=True)
        .size()
        .rename("sample_count")
        .reset_index()
    )
    manifest = {
        "dataset": "zwhe99/DeepMath-103K",
        "revision": "5cf055d1fe3d7a2eb19719ac020211469736ae44",
        "source_rows": len(frame),
        "sample_rows": len(sample),
        "seed": args.seed,
        "stratification": "exact difficulty x full hierarchical topic",
        "allocation": "proportional largest remainder",
        "source_strata": int(len(counts)),
        "sampled_strata": int(len(sampled_counts)),
        "sample_sha256": digest,
        "diagnostic_shards": shard_paths,
        "diagnostic_shard_rows": [
            int((sample["dataset_index"] % 4 == shard_index).sum())
            for shard_index in range(4)
        ],
        "difficulty_counts": {
            str(key): int(value)
            for key, value in sample.groupby("difficulty").size().sort_index().items()
        },
        "topic_counts": {
            str(key): int(value)
            for key, value in sample.groupby("topic").size().sort_index().items()
        },
    }
    write_json(args.output.with_suffix(".manifest.json"), manifest)
    sampled_counts.to_csv(args.output.with_suffix(".strata.csv"), index=False)
    print("DEEPMATH_SAMPLE_OK", json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
