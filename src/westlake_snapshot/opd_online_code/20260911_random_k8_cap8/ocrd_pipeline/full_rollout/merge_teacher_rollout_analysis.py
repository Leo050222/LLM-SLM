#!/usr/bin/env python3
"""Strictly merge teacher or student rollout scores and report statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_rows_sha256(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def quantile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def length_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [int(row[key]) for row in rows]
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p10": quantile(values, 0.10),
        "p25": quantile(values, 0.25),
        "p75": quantile(values, 0.75),
        "p90": quantile(values, 0.90),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
        "max": max(values),
    }


def rollout_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(bool(row["correct"]) for row in rows)
    return {
        "rollout_count": len(rows),
        "correct_count": correct,
        "accuracy": correct / len(rows),
        "finish_reason_counts": dict(sorted(Counter(row["finish_reason"] for row in rows).items())),
        "reached_model_length_count": sum(bool(row["reached_model_length"]) for row in rows),
        "score_method_counts": dict(sorted(Counter(row["score_method"] for row in rows).items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--expected-traces", type=int, default=13_244)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument(
        "--rollout-kind",
        choices=(
            "teacher_full",
            "student_after_bridge",
            "student_from_anchor_prefix",
        ),
        default="teacher_full",
    )
    parser.add_argument(
        "--refresh-completed-summary",
        action="store_true",
        help=(
            "refresh only completion metadata for an already-completed analysis; "
            "all score bytes and all statistics must remain identical"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rollout_kind in {"teacher_full", "student_from_anchor_prefix"}:
        required_marker = "FULL_ROLLOUT_BATCH_COMPLETE"
    elif (args.rollout_root / "GPU_GENERATION_COMPLETE").is_file():
        required_marker = "GPU_GENERATION_COMPLETE"
    else:
        required_marker = "FULL_ROLLOUT_BATCH_COMPLETE"
    final_status = {
        "teacher_full": "TEACHER_ROLLOUT_ANALYSIS_COMPLETE",
        "student_after_bridge": "STUDENT_AFTER_BRIDGE_ROLLOUT_ANALYSIS_COMPLETE",
        "student_from_anchor_prefix": (
            "STUDENT_FROM_ANCHOR_PREFIX_ROLLOUT_ANALYSIS_COMPLETE"
        ),
    }[args.rollout_kind]
    if not (args.rollout_root / required_marker).is_file():
        raise FileNotFoundError(
            f"{args.rollout_kind} required source marker is missing: {required_marker}"
        )
    completion_marker = args.analysis_root / final_status
    if completion_marker.exists() and not args.refresh_completed_summary:
        raise FileExistsError("refusing to overwrite completed analysis")
    if args.refresh_completed_summary and not completion_marker.is_file():
        raise FileNotFoundError(
            "--refresh-completed-summary requires an existing completion marker"
        )

    rows: list[dict[str, Any]] = []
    shard_summaries: list[dict[str, Any]] = []
    for shard_index in range(args.num_shards):
        shard_dir = args.analysis_root / "shards" / f"shard_{shard_index:03d}"
        if not (shard_dir / "SCORE_SHARD_COMPLETE").is_file():
            raise FileNotFoundError(f"shard {shard_index}: marker missing")
        summary = json.loads((shard_dir / "summary.json").read_text(encoding="utf-8"))
        scores_path = shard_dir / "scores.jsonl"
        shard_rows = read_jsonl(scores_path)
        if summary.get("scores_sha256") != sha256(scores_path):
            raise ValueError(f"shard {shard_index}: score SHA mismatch")
        if summary.get("rollout_count") != len(shard_rows):
            raise ValueError(f"shard {shard_index}: score count mismatch")
        rows.extend(shard_rows)
        shard_summaries.append(summary)

    rows.sort(key=lambda row: (int(row["dataset_index"]), int(row["rollout_index"])))
    expected_rollouts = args.expected_traces * args.num_rollouts
    if len(rows) != expected_rollouts:
        raise ValueError(f"expected {expected_rollouts} rollouts, got {len(rows)}")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["dataset_index"])].append(row)
    if len(grouped) != args.expected_traces:
        raise ValueError(f"expected {args.expected_traces} traces, got {len(grouped)}")
    for index, group in grouped.items():
        if [int(row["rollout_index"]) for row in group] != list(range(args.num_rollouts)):
            raise ValueError(f"trace {index}: rollout coverage mismatch")

    correct_rows = [row for row in rows if row["correct"]]
    incorrect_rows = [row for row in rows if not row["correct"]]
    per_trace_correct = [sum(bool(row["correct"]) for row in group) for group in grouped.values()]
    trace_summary = {
        "trace_count": len(grouped),
        "pass_at_8_count": sum(count > 0 for count in per_trace_correct),
        "pass_at_8": sum(count > 0 for count in per_trace_correct) / len(grouped),
        "all_8_correct_count": sum(count == args.num_rollouts for count in per_trace_correct),
        "all_8_correct_rate": sum(count == args.num_rollouts for count in per_trace_correct) / len(grouped),
        "mean_correct_rollouts_per_trace": statistics.fmean(per_trace_correct),
        "correct_rollout_count_histogram": {
            str(key): value for key, value in sorted(Counter(per_trace_correct).items())
        },
    }

    by_difficulty: dict[str, Any] = {}
    difficulty_groups: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        difficulty_groups[float(row["difficulty"])].append(row)
    for difficulty, group in sorted(difficulty_groups.items()):
        by_difficulty[str(difficulty)] = rollout_summary(group)

    by_topic: dict[str, Any] = {}
    topic_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        topic_groups[str(row["topic"])].append(row)
    for topic, group in sorted(topic_groups.items()):
        by_topic[topic] = rollout_summary(group)

    if args.rollout_kind == "teacher_full":
        length_keys = [
            "response_prefix_token_count",
            "teacher_completion_token_count",
            "full_trace_token_count",
        ]
    elif args.rollout_kind == "student_after_bridge":
        length_keys = [
            "response_prefix_token_count",
            "source_bridge_token_count",
            "student_continuation_token_count",
            "completion_token_count",
            "full_trace_token_count",
        ]
    else:
        length_keys = [
            "response_prefix_token_count",
            "student_completion_token_count",
            "completion_token_count",
            "full_trace_token_count",
        ]
    summary = {
        "status": final_status,
        "rollout_kind": args.rollout_kind,
        "source_required_marker": required_marker,
        "source_strict_verification_complete": (
            args.rollout_root / "FULL_ROLLOUT_BATCH_COMPLETE"
        ).is_file(),
        "rollout_root": str(args.rollout_root),
        "analysis_root": str(args.analysis_root),
        "rollout_level": rollout_summary(rows),
        "trace_level": trace_summary,
        "lengths_all": {key: length_summary(rows, key) for key in length_keys},
        "lengths_correct": {key: length_summary(correct_rows, key) for key in length_keys},
        "lengths_incorrect": {key: length_summary(incorrect_rows, key) for key in length_keys},
        "by_difficulty": by_difficulty,
        "by_topic": by_topic,
        "shards": shard_summaries,
    }
    if args.rollout_kind == "student_from_anchor_prefix":
        match_count = sum(
            bool(row["first_token_matches_original_student_anchor"])
            for row in rows
        )
        summary["first_token_matches_original_student_anchor"] = {
            "count": match_count,
            "rate": match_count / len(rows),
        }
    args.analysis_root.mkdir(parents=True, exist_ok=True)
    scores_path = args.analysis_root / "scores.jsonl"
    if args.refresh_completed_summary:
        summary_path = args.analysis_root / "summary.json"
        if not summary_path.is_file() or not scores_path.is_file():
            raise FileNotFoundError(
                "completed analysis is missing summary.json or scores.jsonl"
            )
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        previous_scores_sha = sha256(scores_path)
        if previous.get("scores_sha256") != previous_scores_sha:
            raise ValueError("existing completed scores SHA does not match summary")
        if jsonl_rows_sha256(rows) != previous_scores_sha:
            raise ValueError("reconstructed score bytes differ from completed scores")
        summary["scores_sha256"] = previous_scores_sha
        previous_comparable = dict(previous)
        refreshed_comparable = dict(summary)
        previous_comparable.pop("source_strict_verification_complete", None)
        refreshed_comparable.pop("source_strict_verification_complete", None)
        if previous_comparable != refreshed_comparable:
            raise ValueError(
                "refreshed statistics differ from the completed analysis"
            )
        if summary["source_strict_verification_complete"] is not True:
            raise ValueError("source still lacks strict verification")
        atomic_json(summary_path, summary)
        completion_marker.touch()
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
        return
    atomic_jsonl(scores_path, rows)
    summary["scores_sha256"] = sha256(scores_path)
    atomic_json(args.analysis_root / "summary.json", summary)
    completion_marker.touch()
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
