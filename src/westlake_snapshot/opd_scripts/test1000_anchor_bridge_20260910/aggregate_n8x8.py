#!/usr/bin/env python3
"""Strictly aggregate eight continuation replicates per test1000 bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> str:
    tmp = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            handle.write(payload)
            digest.update(payload.encode())
    os.replace(tmp, path)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    selected_sets = []
    for replicate in range(args.replicates):
        analysis = args.experiment_root / "student_after_bridge_n8x8" / f"replicate_{replicate}" / "analysis"
        if not (analysis / "STUDENT_AFTER_BRIDGE_ROLLOUT_ANALYSIS_COMPLETE").is_file():
            raise FileNotFoundError(f"replicate {replicate} analysis incomplete")
        rows = read_jsonl(analysis / "scores.jsonl")
        selected_sets.append({int(row["dataset_index"]) for row in rows})
        for row in rows:
            key = (int(row["dataset_index"]), int(row["rollout_index"]))
            grouped[key].append({"replicate": replicate, **row})
    if not selected_sets or any(value != selected_sets[0] for value in selected_sets[1:]):
        raise ValueError("replicate trace sets differ")

    bridge_rows = []
    for (dataset_index, bridge_index), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["replicate"]))
        if [int(row["replicate"]) for row in rows] != list(range(args.replicates)):
            raise ValueError(f"incomplete replicate coverage for {(dataset_index, bridge_index)}")
        correct = sum(bool(row["correct"]) for row in rows)
        bridge_rows.append({
            "dataset_index": dataset_index,
            "problem_id": rows[0]["problem_id"],
            "bridge_index": bridge_index,
            "correct_count": correct,
            "avg_at_8": correct / args.replicates,
            "pass_at_8": correct > 0,
            "all_8_correct": correct == args.replicates,
            "continuation_token_count_mean": sum(int(row["completion_token_count"]) for row in rows) / args.replicates,
        })
    expected_bridges = len(selected_sets[0]) * 8
    if len(bridge_rows) != expected_bridges:
        raise ValueError(f"expected {expected_bridges} bridges, got {len(bridge_rows)}")

    by_anchor: dict[int, list[dict]] = defaultdict(list)
    for row in bridge_rows:
        by_anchor[int(row["dataset_index"])].append(row)
    anchor_rows = []
    for dataset_index, rows in sorted(by_anchor.items()):
        rows.sort(key=lambda row: int(row["bridge_index"]))
        if [int(row["bridge_index"]) for row in rows] != list(range(8)):
            raise ValueError(f"anchor {dataset_index} bridge coverage mismatch")
        correct = sum(int(row["correct_count"]) for row in rows)
        anchor_rows.append({
            "dataset_index": dataset_index,
            "problem_id": rows[0]["problem_id"],
            "correct_count_out_of_64": correct,
            "avg_at_64": correct / 64,
            "pass_at_64": correct > 0,
            "all_64_correct": correct == 64,
            "bridges_pass_at_8_count": sum(bool(row["pass_at_8"]) for row in rows),
            "bridges_all_8_correct_count": sum(bool(row["all_8_correct"]) for row in rows),
        })

    bridge_sha = write_jsonl(args.output_dir / "bridge_scores_n8.jsonl", bridge_rows)
    anchor_sha = write_jsonl(args.output_dir / "anchor_scores_n64.jsonl", anchor_rows)
    total = len(bridge_rows) * args.replicates
    correct = sum(int(row["correct_count"]) for row in bridge_rows)
    summary = {
        "status": "TEST1000_ANCHOR_BRIDGE_N8X8_ANALYSIS_COMPLETE",
        "frozen_test_rows": 1000,
        "selected_anchor_count": len(anchor_rows),
        "no_anchor_count": 1000 - len(anchor_rows),
        "bridge_count": len(bridge_rows),
        "continuations_per_bridge": args.replicates,
        "total_continuations": total,
        "correct_continuations": correct,
        "continuation_accuracy_conditional_on_anchor": correct / total,
        "bridge_pass_at_8_count": sum(bool(row["pass_at_8"]) for row in bridge_rows),
        "bridge_pass_at_8": sum(bool(row["pass_at_8"]) for row in bridge_rows) / len(bridge_rows),
        "bridge_all_8_correct_count": sum(bool(row["all_8_correct"]) for row in bridge_rows),
        "bridge_all_8_correct_rate": sum(bool(row["all_8_correct"]) for row in bridge_rows) / len(bridge_rows),
        "anchor_pass_at_64_count": sum(bool(row["pass_at_64"]) for row in anchor_rows),
        "anchor_pass_at_64": sum(bool(row["pass_at_64"]) for row in anchor_rows) / len(anchor_rows),
        "bridge_scores_sha256": bridge_sha,
        "anchor_scores_sha256": anchor_sha,
    }
    tmp = args.output_dir / "summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, args.output_dir / "summary.json")
    (args.output_dir / summary["status"]).touch()
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
