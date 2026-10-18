#!/usr/bin/env python3
"""Strictly verify a complete exact-length bridge batch and write a manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR.parent))

from bridge_generator.run_bridge_batch import (  # noqa: E402
    DEFAULT_TEACHER,
    discover_selected_anchors,
    output_path,
    shard_records,
)


def expected_rows(
    anchor_root: Path, num_shards: int, max_records: int | None
) -> list[tuple[Path, dict[str, Any]]]:
    rows = discover_selected_anchors(anchor_root)
    selected: list[tuple[Path, dict[str, Any]]] = []
    for shard_id in range(num_shards):
        selected.extend(shard_records(rows, shard_id, num_shards, max_records))
    selected.sort(key=lambda item: int(item[1]["dataset_index"]))
    return selected


def validate_artifact(
    value: dict[str, Any], anchor: dict[str, Any], h: int, teacher: str
) -> None:
    index = int(anchor["dataset_index"])
    prefix = [int(token_id) for token_id in anchor["anchor_prefix_token_ids"]]
    student = int(anchor["student_candidate_token_id"])
    bridge = value.get("bridge_token_ids")
    if value.get("status") != "BRIDGE_GENERATED":
        raise ValueError(f"trace {index}: invalid status")
    if int(value.get("dataset_index", -1)) != index:
        raise ValueError(f"trace {index}: dataset_index mismatch")
    if value.get("trace_sha256") != anchor.get("trace_sha256"):
        raise ValueError(f"trace {index}: trace_sha256 mismatch")
    if value.get("teacher_model") != teacher:
        raise ValueError(f"trace {index}: teacher mismatch")
    if value.get("requested_bridge_length") != h:
        raise ValueError(f"trace {index}: requested length mismatch")
    if value.get("actual_bridge_length") != h:
        raise ValueError(f"trace {index}: actual length mismatch")
    if not isinstance(bridge, list) or len(bridge) != h:
        raise ValueError(f"trace {index}: bridge token count mismatch")
    if not all(isinstance(token_id, int) and token_id >= 0 for token_id in bridge):
        raise ValueError(f"trace {index}: invalid bridge token IDs")
    if bridge[0] == student:
        raise ValueError(f"trace {index}: first bridge token equals student token")
    if value.get("first_token_differs_from_student") is not True:
        raise ValueError(f"trace {index}: first-token flag is false")
    if value.get("bridge_first_token_id") != bridge[0]:
        raise ValueError(f"trace {index}: bridge_first_token_id mismatch")
    if value.get("anchor_prefix_token_count") != len(prefix):
        raise ValueError(f"trace {index}: prefix count mismatch")
    if value.get("bridge_prefix_token_ids") != prefix + bridge:
        raise ValueError(f"trace {index}: resumable bridge prefix mismatch")
    if len(value.get("bridge_token_texts", [])) != h:
        raise ValueError(f"trace {index}: bridge token texts mismatch")
    if len(value.get("bridge_chosen_logprobs", [])) != h:
        raise ValueError(f"trace {index}: chosen logprobs mismatch")
    first_step = value.get("first_step", {})
    if first_step.get("chosen_token_id") != bridge[0]:
        raise ValueError(f"trace {index}: first-step chosen token mismatch")


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--h", type=int, default=256)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--teacher", default=DEFAULT_TEACHER)
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()

    rows = expected_rows(args.anchor_root, args.num_shards, args.max_records)
    if args.expected_count is not None and len(rows) != args.expected_count:
        raise ValueError(f"expected rows {len(rows)} != {args.expected_count}")
    for shard_id in range(args.num_shards):
        marker = args.output_root / "shards" / f"shard_{shard_id:02d}" / "SHARD_COMPLETE"
        if not marker.is_file():
            raise FileNotFoundError(f"missing shard marker: {marker}")

    expected_paths: set[Path] = set()
    manifest_rows: list[dict[str, Any]] = []
    for anchor_path, anchor in rows:
        index = int(anchor["dataset_index"])
        path = output_path(args.output_root, index, args.h, args.num_rollouts)
        expected_paths.add(path.resolve())
        bundle = json.loads(path.read_text(encoding="utf-8"))
        rollouts = bundle.get("rollouts")
        if not bool(
            bundle.get("status") == "BRIDGE_ROLLOUTS_GENERATED"
            and bundle.get("dataset_index") == index
            and bundle.get("trace_sha256") == anchor.get("trace_sha256")
            and bundle.get("teacher_model") == args.teacher
            and bundle.get("requested_bridge_length") == args.h
            and bundle.get("requested_rollouts") == args.num_rollouts
            and bundle.get("actual_rollouts") == args.num_rollouts
            and bundle.get("all_first_tokens_differ_from_student") is True
            and isinstance(rollouts, list)
            and len(rollouts) == args.num_rollouts
        ):
            raise ValueError(f"trace {index}: invalid rollout bundle")
        for rollout_index, rollout in enumerate(rollouts):
            if rollout.get("rollout_index") != rollout_index:
                raise ValueError(f"trace {index}: rollout index/order mismatch")
            validate_artifact(rollout, anchor, args.h, args.teacher)
        manifest_rows.append(
            {
                "dataset_index": index,
                "problem_id": anchor.get("problem_id"),
                "trace_sha256": anchor.get("trace_sha256"),
                "anchor_artifact": str(anchor_path),
                "bridge_artifact": str(path),
                "anchor_response_position": anchor.get("anchor_response_position"),
                "anchor_prefix_token_count": len(anchor["anchor_prefix_token_ids"]),
                "student_anchor_token_id": anchor["student_candidate_token_id"],
                "bridge_rollout_count": len(rollouts),
                "bridge_first_token_ids": [
                    rollout["bridge_first_token_id"] for rollout in rollouts
                ],
                "bridge_length": args.h,
            }
        )

    actual_paths = {
        path.resolve()
        for path in args.output_root.glob(
            f"traces/trace_*/bridge_h{args.h}_n{args.num_rollouts}.json"
        )
    }
    missing = expected_paths - actual_paths
    extras = actual_paths - expected_paths
    if missing or extras:
        raise ValueError(f"artifact set mismatch: missing={len(missing)} extras={len(extras)}")

    manifest_path = args.output_root / "manifest.jsonl"
    temporary = manifest_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    summary = {
        "status": "BRIDGE_BATCH_COMPLETE",
        "verified_trace_count": len(rows),
        "verified_bridge_rollout_count": len(rows) * args.num_rollouts,
        "requested_bridge_length": args.h,
        "requested_rollouts_per_trace": args.num_rollouts,
        "teacher_model": args.teacher,
        "num_shards": args.num_shards,
        "first_token_differs_count": len(rows) * args.num_rollouts,
        "missing_artifact_count": 0,
        "extra_artifact_count": 0,
    }
    atomic_write_json(args.output_root / "summary.json", summary)
    (args.output_root / "BRIDGE_BATCH_COMPLETE").touch()
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
