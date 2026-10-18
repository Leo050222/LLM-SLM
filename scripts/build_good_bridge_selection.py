#!/usr/bin/env python3
"""Build a strict training selection where bridge 8-rollout accuracy improves over direct Student."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


EXPECTED_TRACES = 13_244
BRIDGES_PER_TRACE = 8
EXPECTED_BRIDGES = EXPECTED_TRACES * BRIDGES_PER_TRACE
EXPECTED_GOOD = 37_724


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-analysis", type=Path, required=True)
    parser.add_argument("--bridge-aggregate", type=Path, required=True)
    parser.add_argument("--bridge-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    direct_scores = args.direct_analysis / "scores.jsonl"
    direct_summary = read_json(args.direct_analysis / "summary.json")
    direct_marker = args.direct_analysis / "STUDENT_FROM_ANCHOR_PREFIX_ROLLOUT_ANALYSIS_COMPLETE"
    if not direct_marker.is_file():
        raise FileNotFoundError(direct_marker)
    if direct_summary.get("status") != direct_marker.name:
        raise RuntimeError("direct Student analysis status mismatch")
    if direct_summary.get("scores_sha256") != sha256(direct_scores):
        raise RuntimeError("direct Student score SHA mismatch")

    aggregate_scores = args.bridge_aggregate / "bridge_scores_x8.jsonl"
    aggregate_summary = read_json(args.bridge_aggregate / "summary.json")
    aggregate_marker = args.bridge_aggregate / "STUDENT_AFTER_BRIDGE_X8_AGGREGATE_COMPLETE"
    if not aggregate_marker.is_file():
        raise FileNotFoundError(aggregate_marker)
    if aggregate_summary.get("status") != aggregate_marker.name:
        raise RuntimeError("bridge aggregate status mismatch")
    if aggregate_summary.get("scores_sha256") != sha256(aggregate_scores):
        raise RuntimeError("bridge aggregate score SHA mismatch")
    if int(aggregate_summary.get("bridge_count", -1)) != EXPECTED_BRIDGES:
        raise RuntimeError("bridge aggregate count mismatch")
    if int(aggregate_summary.get("replicates_per_bridge", -1)) != 8:
        raise RuntimeError("bridge replicate count mismatch")

    bridge_manifest = args.bridge_root / "manifest.jsonl"
    bridge_marker = args.bridge_root / "BRIDGE_BATCH_COMPLETE"
    if not bridge_marker.is_file():
        raise FileNotFoundError(bridge_marker)
    manifest_rows = [json.loads(line) for line in bridge_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest_rows.sort(key=lambda row: int(row["dataset_index"]))
    if len(manifest_rows) != EXPECTED_TRACES:
        raise RuntimeError("bridge manifest trace count mismatch")
    dataset_to_position = {int(row["dataset_index"]): position for position, row in enumerate(manifest_rows)}
    if len(dataset_to_position) != EXPECTED_TRACES:
        raise RuntimeError("duplicate bridge manifest dataset indices")

    direct_correct: dict[int, int] = defaultdict(int)
    direct_seen: dict[int, set[int]] = defaultdict(set)
    direct_rows = 0
    with direct_scores.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            dataset_index = int(row["dataset_index"])
            rollout_index = int(row["rollout_index"])
            direct_rows += 1
            if rollout_index in direct_seen[dataset_index]:
                raise RuntimeError(f"duplicate direct rollout: {dataset_index}/{rollout_index}")
            direct_seen[dataset_index].add(rollout_index)
            direct_correct[dataset_index] += int(bool(row["correct"]))
    if direct_rows != EXPECTED_BRIDGES or set(direct_seen) != set(dataset_to_position):
        raise RuntimeError("direct Student trace coverage mismatch")
    if any(indices != set(range(8)) for indices in direct_seen.values()):
        raise RuntimeError("direct Student rollout-index coverage mismatch")

    records = []
    aggregate_keys = set()
    with aggregate_scores.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            dataset_index = int(row["dataset_index"])
            bridge_index = int(row["bridge_index"])
            key = (dataset_index, bridge_index)
            if key in aggregate_keys:
                raise RuntimeError(f"duplicate aggregate bridge key: {key}")
            aggregate_keys.add(key)
            if dataset_index not in dataset_to_position or not 0 <= bridge_index < 8:
                raise RuntimeError(f"invalid aggregate bridge key: {key}")
            before = direct_correct[dataset_index]
            after = int(row["correct_count"])
            if not 0 <= after <= 8:
                raise RuntimeError(f"invalid aggregate correct count: {key}")
            if after > before:
                sample_id = dataset_to_position[dataset_index] * BRIDGES_PER_TRACE + bridge_index
                records.append({
                    "sample_id": sample_id,
                    "dataset_index": dataset_index,
                    "bridge_index": bridge_index,
                    "direct_correct_count": before,
                    "after_bridge_correct_count": after,
                    "correct_count_lift": after - before,
                    "accuracy_lift": (after - before) / 8,
                })
    if len(aggregate_keys) != EXPECTED_BRIDGES:
        raise RuntimeError("aggregate bridge key coverage mismatch")
    records.sort(key=lambda row: int(row["sample_id"]))
    sample_ids = [int(row["sample_id"]) for row in records]
    if len(records) != EXPECTED_GOOD or len(set(sample_ids)) != EXPECTED_GOOD:
        raise RuntimeError(f"expected {EXPECTED_GOOD} good bridges, got {len(records)}")

    args.output_root.mkdir(parents=True, exist_ok=False)
    selection = {
        "status": "GOOD_BRIDGE_SELECTION_COMPLETE",
        "definition": "after_bridge_correct_count_out_of_8 > direct_student_correct_count_out_of_8_for_same_prefix",
        "strict_comparison": True,
        "trace_count": EXPECTED_TRACES,
        "source_bridge_count": EXPECTED_BRIDGES,
        "good_bridge_count": len(records),
        "good_bridge_fraction": len(records) / EXPECTED_BRIDGES,
        "replicates_per_condition": 8,
        "bridge_manifest": str(bridge_manifest),
        "bridge_manifest_sha256": sha256(bridge_manifest),
        "direct_scores": str(direct_scores),
        "direct_scores_sha256": sha256(direct_scores),
        "aggregate_scores": str(aggregate_scores),
        "aggregate_scores_sha256": sha256(aggregate_scores),
        "sample_ids": sample_ids,
        "records": records,
    }
    output = args.output_root / "selection.json"
    atomic_json(output, selection)
    (args.output_root / selection["status"]).touch()
    print(json.dumps({
        "status": selection["status"],
        "output": str(output),
        "selection_sha256": sha256(output),
        "good_bridge_count": len(records),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
