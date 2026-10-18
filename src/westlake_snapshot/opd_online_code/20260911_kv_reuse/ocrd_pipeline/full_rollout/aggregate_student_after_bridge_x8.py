#!/usr/bin/env python3
"""Strictly combine the original and seven added samples per bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


SOURCE_STATUS = "FULL_ROLLOUT_BATCH_COMPLETE"
ANALYSIS_STATUS = "STUDENT_AFTER_BRIDGE_ROLLOUT_ANALYSIS_COMPLETE"
EXPECTED_TRACES = 13_244
BRIDGES_PER_TRACE = 8
REPLICATES = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_scores(
    root: Path, bridge_token_limit: int | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_summary = read_json(root / "summary.json")
    if source_summary.get("status") != SOURCE_STATUS:
        raise ValueError(f"{root}: source status mismatch")
    if source_summary.get("verified_trace_count") != EXPECTED_TRACES:
        raise ValueError(f"{root}: source trace count mismatch")
    if source_summary.get("verified_rollout_count") != EXPECTED_TRACES * BRIDGES_PER_TRACE:
        raise ValueError(f"{root}: source rollout count mismatch")
    if source_summary.get("source_bridge_token_limit") != bridge_token_limit:
        raise ValueError(f"{root}: source bridge token limit mismatch")
    if not (root / SOURCE_STATUS).is_file():
        raise FileNotFoundError(f"{root}: strict source marker missing")
    analysis = root / "analysis_correctness_length_single_job"
    analysis_summary = read_json(analysis / "summary.json")
    scores_path = analysis / "scores.jsonl"
    if analysis_summary.get("status") != ANALYSIS_STATUS:
        raise ValueError(f"{root}: analysis status mismatch")
    if analysis_summary.get("source_strict_verification_complete") is not True:
        raise ValueError(f"{root}: analysis lacks strict source verification")
    if analysis_summary.get("scores_sha256") != sha256(scores_path):
        raise ValueError(f"{root}: score SHA mismatch")
    if not (analysis / ANALYSIS_STATUS).is_file():
        raise FileNotFoundError(f"{root}: analysis marker missing")
    rows = [json.loads(line) for line in scores_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda row: (int(row["dataset_index"]), int(row["rollout_index"])))
    expected = EXPECTED_TRACES * BRIDGES_PER_TRACE
    if len(rows) != expected:
        raise ValueError(f"{root}: expected {expected} scores, got {len(rows)}")
    keys = [(int(row["dataset_index"]), int(row["rollout_index"])) for row in rows]
    if len(set(keys)) != expected:
        raise ValueError(f"{root}: duplicate bridge keys")
    bridges_by_trace: dict[int, list[int]] = {}
    for dataset_index, bridge_index in keys:
        bridges_by_trace.setdefault(dataset_index, []).append(bridge_index)
    if len(bridges_by_trace) != EXPECTED_TRACES:
        raise ValueError(
            f"{root}: expected {EXPECTED_TRACES} sparse dataset indices, "
            f"got {len(bridges_by_trace)}"
        )
    expected_bridges = list(range(BRIDGES_PER_TRACE))
    for dataset_index, bridge_indices in bridges_by_trace.items():
        if bridge_indices != expected_bridges:
            raise ValueError(
                f"{root}: dataset index {dataset_index} bridge coverage mismatch: "
                f"{bridge_indices}"
            )
    return rows, {"root": str(root), "scores_sha256": sha256(scores_path), "correct_count": sum(bool(row["correct"]) for row in rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--replicate-parent", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bridge-token-limit", type=int)
    args = parser.parse_args()
    marker = args.output_root / "STUDENT_AFTER_BRIDGE_X8_AGGREGATE_COMPLETE"
    if marker.exists():
        raise FileExistsError(f"refusing to overwrite completed aggregate: {marker}")

    roots = [args.original_root] + [args.replicate_parent / f"replicate_{index:02d}" for index in range(1, REPLICATES)]
    all_rows: list[list[dict[str, Any]]] = []
    sources: list[dict[str, Any]] = []
    for root in roots:
        rows, source = load_scores(root, args.bridge_token_limit)
        all_rows.append(rows)
        sources.append(source)

    args.output_root.mkdir(parents=True, exist_ok=True)
    scores_path = args.output_root / "bridge_scores_x8.jsonl"
    temporary = scores_path.with_suffix(".jsonl.tmp")
    histogram: Counter[int] = Counter()
    total_correct = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for position in range(EXPECTED_TRACES * BRIDGES_PER_TRACE):
            reference = all_rows[0][position]
            correctness = [bool(rows[position]["correct"]) for rows in all_rows]
            for replicate_rows in all_rows[1:]:
                candidate = replicate_rows[position]
                if (candidate["dataset_index"], candidate["rollout_index"], candidate["problem_id"], candidate["target_answer"]) != (reference["dataset_index"], reference["rollout_index"], reference["problem_id"], reference["target_answer"]):
                    raise ValueError(f"replicate alignment mismatch at row {position}")
            correct_count = sum(correctness)
            total_correct += correct_count
            histogram[correct_count] += 1
            output = {
                "dataset_index": int(reference["dataset_index"]),
                "problem_id": reference["problem_id"],
                "bridge_index": int(reference["rollout_index"]),
                "correct_by_replicate": correctness,
                "correct_count": correct_count,
                "success_rate": correct_count / REPLICATES,
                "difficulty": reference["difficulty"],
                "topic": reference["topic"],
            }
            handle.write(json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, scores_path)

    bridge_count = EXPECTED_TRACES * BRIDGES_PER_TRACE
    continuation_count = bridge_count * REPLICATES
    summary = {
        "status": "STUDENT_AFTER_BRIDGE_X8_AGGREGATE_COMPLETE",
        "trace_count": EXPECTED_TRACES,
        "bridge_count": bridge_count,
        "replicates_per_bridge": REPLICATES,
        "source_bridge_token_limit": args.bridge_token_limit,
        "continuation_count": continuation_count,
        "correct_count": total_correct,
        "accuracy": total_correct / continuation_count,
        "pass_at_8_count": bridge_count - histogram[0],
        "pass_at_8": (bridge_count - histogram[0]) / bridge_count,
        "all_8_correct_count": histogram[REPLICATES],
        "correct_count_histogram": {str(index): histogram[index] for index in range(REPLICATES + 1)},
        "scores_sha256": sha256(scores_path),
        "sources": sources,
    }
    atomic_json(args.output_root / "summary.json", summary)
    marker.touch()
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
