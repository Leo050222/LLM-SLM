#!/usr/bin/env python3
"""Merge and verify sharded DeepMath GE6 baseline generations."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


KEY_FIELDS = (
    "dataset_index",
    "problem_id",
    "problem",
    "solution",
    "difficulty",
    "topic",
    "source_index",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-data", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--expected-seed", type=int, required=True)
    parser.add_argument("--expected-max-tokens", type=int, required=True)
    parser.add_argument("--expected-max-model-len", type=int, required=True)
    args = parser.parse_args()

    if sha256(args.full_data) != args.expected_data_sha256:
        raise RuntimeError("Full-data SHA mismatch")
    raw_source_rows = pq.read_table(args.full_data).to_pylist()
    if len(raw_source_rows) != 1000:
        raise RuntimeError(f"Expected 1000 source rows, got {len(raw_source_rows)}")
    source: dict[int, dict[str, Any]] = {}
    for dataset_index, row in enumerate(raw_source_rows):
        extra = row["extra_info"]
        reward = row["reward_model"]
        source[dataset_index] = {
            "dataset_index": dataset_index,
            "problem_id": str(extra["problem_id"]),
            "problem": str(extra["question"]),
            "solution": str(reward["ground_truth"]),
            "difficulty": float(extra["difficulty"]),
            "topic": str(extra["topic"]),
            "source_index": int(extra["source_index"]),
        }

    shard_dirs = sorted(args.shard_root.glob("shard_*"))
    if len(shard_dirs) != 4:
        raise RuntimeError(f"Expected four shard directories, got {len(shard_dirs)}")
    rows: list[dict[str, Any]] = []
    elapsed = 0.0
    for shard_index, shard_dir in enumerate(shard_dirs):
        config = read_json(shard_dir / "run_config.json")
        sampling = config["sampling"]
        expected = {
            "student": args.expected_model,
            "num_prompts": 250,
            "max_model_len": args.expected_max_model_len,
            "thinking_mode": True,
            "parameters_frozen": True,
            "optimizer_created": False,
            "backward_called": False,
        }
        for key, value in expected.items():
            if config.get(key) != value:
                raise RuntimeError(f"{shard_dir}: {key}={config.get(key)!r}, expected {value!r}")
        sampling_expected = {
            "max_tokens": args.expected_max_tokens,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "n": 1,
            "seed": args.expected_seed,
        }
        for key, value in sampling_expected.items():
            if sampling.get(key) != value:
                raise RuntimeError(f"{shard_dir}: sampling.{key} mismatch")
        shard_rows = read_jsonl(shard_dir / "rollouts.jsonl")
        if len(shard_rows) != 250:
            raise RuntimeError(f"{shard_dir}: expected 250 rows, got {len(shard_rows)}")
        if any(int(row["dataset_index"]) % 4 != shard_index for row in shard_rows):
            raise RuntimeError(f"{shard_dir}: modulo-shard membership mismatch")
        rows.extend(shard_rows)
        elapsed += float(config.get("elapsed_seconds_generate", 0.0))

    rows.sort(key=lambda row: int(row["dataset_index"]))
    if [int(row["dataset_index"]) for row in rows] != list(range(1000)):
        raise RuntimeError("Merged rollout indices are incomplete, duplicated, or out of order")
    for row in rows:
        reference = source[int(row["dataset_index"])]
        for field in KEY_FIELDS:
            left = row.get(field)
            right = reference.get(field)
            if field == "difficulty":
                left, right = float(left), float(right)
            elif field in ("dataset_index", "source_index"):
                left, right = int(left), int(right)
            elif field in ("problem_id", "problem", "solution", "topic"):
                left, right = str(left), str(right)
            if left != right:
                raise RuntimeError(f"index={row['dataset_index']}: metadata mismatch in {field}")

    lengths = [int(row["response_tokens"]) for row in rows]
    correct = sum(bool(row["correct"]) for row in rows)
    by_difficulty: dict[str, dict[str, Any]] = {}
    groups: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[float(row["difficulty"])].append(row)
    for difficulty, group in sorted(groups.items()):
        group_correct = sum(bool(row["correct"]) for row in group)
        by_difficulty[str(difficulty)] = {
            "rows": len(group),
            "correct": group_correct,
            "accuracy": group_correct / len(group),
            "mean_response_tokens": sum(int(row["response_tokens"]) for row in group) / len(group),
        }

    summary = {
        "status": "DEEPMATH_GE6_QWEN3_1_7B_BASELINE_OK",
        "model": args.expected_model,
        "data": str(args.full_data),
        "data_sha256": args.expected_data_sha256,
        "rows": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "total_response_tokens": sum(lengths),
        "mean_response_tokens": sum(lengths) / len(lengths),
        "max_response_tokens_observed": max(lengths),
        "finish_reason": dict(sorted(Counter(str(row["finish_reason"]) for row in rows).items())),
        "thinking_open_tag": sum("<think>" in str(row["response"]) for row in rows),
        "thinking_close_tag": sum("</think>" in str(row["response"]) for row in rows),
        "sampling": {
            "thinking": True,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "seed": args.expected_seed,
            "max_response_tokens": args.expected_max_tokens,
            "max_model_len": args.expected_max_model_len,
        },
        "elapsed_seconds_generate_sum_shards": elapsed,
        "by_difficulty": by_difficulty,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "rollouts.jsonl", rows)
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "BASELINE_COMPLETE").touch()
    print("DEEPMATH_GE6_QWEN3_1_7B_BASELINE_OK")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
