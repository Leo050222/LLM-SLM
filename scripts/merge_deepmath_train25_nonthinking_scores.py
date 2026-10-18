#!/usr/bin/env python3
"""Merge and strictly verify rule scores for all saved rollout traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPECTED_ROWS = 14_208
EXPECTED_UNIQUE = 14_158
EXPECTED_PADDING = 50
NUM_SHARDS = 16
ROWS_PER_SHARD = 888
SOURCE_SHA256 = "fdcb58732d22f5ab27d111f07c8746836d983c5d15d1d800970154ea1f3cfc91"


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


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def wilson(correct: int, total: int, z: float = 1.959963984540054) -> list[float]:
    rate = correct / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return [center - radius, center + radius]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(bool(row["correct"]) for row in rows)
    return {
        "rows": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "accuracy_percent": 100 * correct / len(rows),
        "wilson_95": wilson(correct, len(rows)),
        "score_methods": dict(sorted(Counter(str(row["score_method"]) for row in rows).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--heldout-same-protocol-accuracy", type=float, required=True)
    args = parser.parse_args()

    if not 0.0 <= args.heldout_same_protocol_accuracy <= 1.0:
        raise ValueError("held-out accuracy must be in [0, 1]")

    summary_path = args.score_root / "summary.json"
    if summary_path.exists() or (args.score_root / "SCORING_COMPLETE").exists():
        raise FileExistsError("Refusing to overwrite completed scoring output")
    rollout_summary = read_json(args.run_root / "summary.json")
    if rollout_summary.get("status") != "DEEPMATH_TRAIN25_NATIVE_NONTHINKING_ROLLOUT_OK":
        raise RuntimeError("Rollout summary status mismatch")
    if rollout_summary.get("source_sha256") != SOURCE_SHA256:
        raise RuntimeError("Rollout source SHA mismatch")

    merged: list[dict[str, Any]] = []
    shard_summaries: list[dict[str, Any]] = []
    for shard_index in range(NUM_SHARDS):
        trace_dir = args.run_root / "shards" / f"shard_{shard_index:02d}"
        score_dir = args.score_root / "shards" / f"shard_{shard_index:02d}"
        if not (score_dir / "SCORE_SHARD_COMPLETE").is_file():
            raise RuntimeError(f"Shard {shard_index}: score marker missing")
        traces = read_jsonl(trace_dir / "rollouts.jsonl")
        scores = read_jsonl(score_dir / "scores.jsonl")
        config = read_json(score_dir / "score_config.json")
        if len(traces) != ROWS_PER_SHARD or len(scores) != ROWS_PER_SHARD:
            raise RuntimeError(f"Shard {shard_index}: row count mismatch")
        expected_config = {
            "status": "DEEPMATH_TRAIN25_NATIVE_NONTHINKING_SCORE_SHARD_OK",
            "rollouts_sha256": sha256(trace_dir / "rollouts.jsonl"),
            "scores_sha256": sha256(score_dir / "scores.jsonl"),
            "scorer": "run_opd_logit_diagnostic.score_math_response",
            "score_timeout_seconds": 20.0,
            "shard_index": shard_index,
            "num_shards": NUM_SHARDS,
            "rows": ROWS_PER_SHARD,
        }
        for key, expected in expected_config.items():
            if config.get(key) != expected:
                raise RuntimeError(f"Shard {shard_index}: config {key} mismatch")
        for trace, score in zip(traces, scores, strict=True):
            index = int(trace["dataset_index"])
            if int(score["dataset_index"]) != index:
                raise RuntimeError(f"Shard {shard_index}: score order mismatch")
            if str(score["problem_id"]) != str(trace["problem_id"]):
                raise RuntimeError(f"index={index}: problem_id mismatch")
            if str(score["target_answer"]) != str(trace["solution"]):
                raise RuntimeError(f"index={index}: target mismatch")
            merged.append(
                {
                    **score,
                    "difficulty": float(trace["difficulty"]),
                    "topic": str(trace["topic"]),
                    "source_index": int(trace["source_index"]),
                    "padded_duplicate": bool(trace["padded_duplicate"]),
                    "response_tokens": int(trace["response_tokens"]),
                    "finish_reason": trace["finish_reason"],
                }
            )
        shard_summaries.append(config)

    merged.sort(key=lambda row: int(row["dataset_index"]))
    if [int(row["dataset_index"]) for row in merged] != list(range(EXPECTED_ROWS)):
        raise RuntimeError("Global score coverage is incomplete or duplicated")
    unique_rows = [row for row in merged if not bool(row["padded_duplicate"])]
    padded_rows = [row for row in merged if bool(row["padded_duplicate"])]
    if len(unique_rows) != EXPECTED_UNIQUE or len(padded_rows) != EXPECTED_PADDING:
        raise RuntimeError("Unique/padding partition mismatch")

    by_difficulty: dict[str, Any] = {}
    groups: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in unique_rows:
        groups[float(row["difficulty"])].append(row)
    for difficulty, rows in sorted(groups.items()):
        by_difficulty[str(difficulty)] = aggregate(rows)

    unique_summary = aggregate(unique_rows)
    all_summary = aggregate(merged)
    summary = {
        "status": "DEEPMATH_TRAIN25_NATIVE_NONTHINKING_SCORING_OK",
        "run_root": str(args.run_root),
        "source_sha256": SOURCE_SHA256,
        "model": rollout_summary.get("model"),
        "tokenizer": rollout_summary.get("tokenizer"),
        "scorer": "run_opd_logit_diagnostic.score_math_response",
        "score_timeout_seconds": 20.0,
        "unique": unique_summary,
        "all_rows_including_padding": all_summary,
        "padding_only": aggregate(padded_rows),
        "by_difficulty_unique": by_difficulty,
        "heldout_same_protocol_accuracy": args.heldout_same_protocol_accuracy,
        "unique_delta_vs_heldout_percentage_points": 100
        * (unique_summary["accuracy"] - args.heldout_same_protocol_accuracy),
        "comparison_note": "Descriptive only: training split and held-out set contain different problems.",
        "shards": shard_summaries,
    }
    args.score_root.mkdir(parents=True, exist_ok=True)
    merged_path = args.score_root / "scores.jsonl"
    atomic_jsonl(merged_path, merged)
    summary["scores_sha256"] = sha256(merged_path)
    atomic_json(summary_path, summary)
    (args.score_root / "SCORING_COMPLETE").touch()
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    print("DEEPMATH_TRAIN25_NATIVE_NONTHINKING_SCORING_OK")


if __name__ == "__main__":
    main()
