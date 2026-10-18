#!/usr/bin/env python3
"""Merge and verify the four Student-Top16 trace diagnostic shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    dirs = sorted(args.root.glob("shard_*"))
    if len(dirs) != 4:
        raise RuntimeError(f"Expected four shards, found {len(dirs)}")
    token_paths = [path for directory in dirs for path in sorted((directory / "token_shards").glob("example_*.parquet"))]
    detail_paths = [path for directory in dirs for path in sorted((directory / "top16_details").glob("example_*.npz"))]
    if len(token_paths) != 209 or len(detail_paths) != 209:
        raise RuntimeError(f"Expected 209 token/detail shards, found {len(token_paths)}/{len(detail_paths)}")
    tokens = pd.concat((pd.read_parquet(path) for path in token_paths), ignore_index=True)
    if tokens["dataset_index"].nunique() != 209:
        raise RuntimeError("Dataset indices are incomplete or duplicated")
    numeric = tokens.select_dtypes(include="number")
    if numeric.isna().any().any():
        raise RuntimeError("Numeric output contains NaN")
    metrics = [
        "student_top16_renorm_kl", "student_teacher_top16_overlap", "top1_agreement",
        "student_sample_logprob", "teacher_sample_logprob",
        "sample_logprob_gap_student_minus_teacher", "sampled_reverse_kl_k3",
    ]
    per_rollout = tokens.groupby("dataset_index", observed=True)[metrics].mean().reset_index()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokens.to_parquet(args.output_dir / "tokens.parquet", index=False, compression="zstd")
    per_rollout.to_parquet(args.output_dir / "per_rollout.parquet", index=False, compression="zstd")
    summary = {
        "status": "STUDENT_TOP16_TRACE_MERGE_OK",
        "topk": 16,
        "support": "student-selected; stop-gradient; renormalized",
        "selected_rollouts": 209,
        "tokens": int(len(tokens)),
        "token_weighted_mean": {name: float(tokens[name].mean()) for name in metrics},
        "rollout_weighted_mean": {name: float(per_rollout[name].mean()) for name in metrics},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
