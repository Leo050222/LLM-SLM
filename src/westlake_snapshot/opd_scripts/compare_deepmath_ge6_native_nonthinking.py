#!/usr/bin/env python3
"""Verify alignment and compare the 1.7B and 14B non-thinking runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def pair_label(small_ok: bool, large_ok: bool) -> str:
    if small_ok and large_ok:
        return "both_correct"
    if small_ok:
        return "qwen3_1_7b_only_correct"
    if large_ok:
        return "qwen3_14b_only_correct"
    return "both_wrong"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen17-dir", type=Path, required=True)
    parser.add_argument("--qwen14-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {args.output_dir}")

    small_summary = read_json(args.qwen17_dir / "summary.json")
    large_summary = read_json(args.qwen14_dir / "summary.json")
    for summary in (small_summary, large_summary):
        if summary.get("status") != "DEEPMATH_GE6_NATIVE_NONTHINKING_OK":
            raise RuntimeError("Input summary status is not complete")
        if summary.get("rows") != 1000 or summary["sampling"].get("thinking") is not False:
            raise RuntimeError("Input summary protocol mismatch")
        if summary.get("thinking_open_tag") != 0 or summary.get("thinking_close_tag") != 0:
            raise RuntimeError("Thinking tags were observed")
    invariant_keys = ("tokenizer", "data_sha256", "rows", "sampling")
    for key in invariant_keys:
        if small_summary.get(key) != large_summary.get(key):
            raise RuntimeError(f"Cross-model protocol mismatch: {key}")

    small = read_jsonl(args.qwen17_dir / "rollouts.jsonl")
    large = read_jsonl(args.qwen14_dir / "rollouts.jsonl")
    small.sort(key=lambda row: int(row["dataset_index"]))
    large.sort(key=lambda row: int(row["dataset_index"]))
    small_keys = [(int(row["dataset_index"]), str(row["problem_id"])) for row in small]
    large_keys = [(int(row["dataset_index"]), str(row["problem_id"])) for row in large]
    if small_keys != large_keys or len(small_keys) != 1000:
        raise RuntimeError("The two model outputs do not align on all 1000 examples")

    paired = Counter()
    by_difficulty: dict[float, Counter[str]] = defaultdict(Counter)
    for small_row, large_row in zip(small, large, strict=True):
        label = pair_label(bool(small_row["correct"]), bool(large_row["correct"]))
        paired[label] += 1
        by_difficulty[float(small_row["difficulty"])][label] += 1
    comparison = {
        "status": "DEEPMATH_GE6_NATIVE_NONTHINKING_COMPARISON_OK",
        "qwen3_1_7b": small_summary,
        "qwen3_14b": large_summary,
        "accuracy_delta_14b_minus_1_7b": large_summary["accuracy"] - small_summary["accuracy"],
        "paired": dict(sorted(paired.items())),
        "paired_by_difficulty": {
            str(key): dict(sorted(value.items())) for key, value in sorted(by_difficulty.items())
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "comparison.json", comparison)
    (args.output_dir / "COMPARISON_COMPLETE").touch()
    print("DEEPMATH_GE6_NATIVE_NONTHINKING_COMPARISON_OK")
    print(json.dumps(comparison, sort_keys=True))


if __name__ == "__main__":
    main()
