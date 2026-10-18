#!/usr/bin/env python3
"""Read-only verification for the DeepMath max-response evaluation."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


MODELS = {
    "qwen3_1_7b": ("merged_qwen3_1_7b", "qwen3_1_7b_rollouts.jsonl"),
    "qwen3_4b": ("merged", "qwen3_4b_rollouts.jsonl"),
    "qwen3_14b": ("merged", "qwen3_14b_rollouts.jsonl"),
}
EXPECTED_ROWS = 1000
EXPECTED_MAX_RESPONSE = 40512
EXPECTED_MAX_MODEL_LEN = 40960
KEY_FIELDS = (
    "dataset_index",
    "problem_id",
    "problem",
    "difficulty",
    "topic",
    "source_index",
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != EXPECTED_ROWS:
        raise AssertionError(f"{path}: expected {EXPECTED_ROWS} rows, got {len(rows)}")
    return rows


def keyed(rows: list[dict[str, Any]], path: Path) -> dict[int, dict[str, Any]]:
    result = {int(row["dataset_index"]): row for row in rows}
    if len(result) != EXPECTED_ROWS or set(result) != set(range(EXPECTED_ROWS)):
        raise AssertionError(f"{path}: dataset_index is not exactly 0..999")
    return result


def assert_finite_tree(value: Any, label: str) -> None:
    if isinstance(value, dict):
        if "count" in value and "finite_count" in value:
            if int(value["count"]) != int(value["finite_count"]):
                raise AssertionError(f"{label}: finite_count != count: {value}")
        for key, child in value.items():
            assert_finite_tree(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite_tree(child, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise AssertionError(f"{label}: non-finite value {value!r}")


def model_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(bool(row["correct"]) for row in rows)
    lengths = [int(row["response_tokens"]) for row in rows]
    return {
        "correct": correct,
        "accuracy": correct / len(rows),
        "mean_response_tokens": sum(lengths) / len(lengths),
        "max_response_tokens_observed": max(lengths),
        "finish_reason": dict(sorted(Counter(row["finish_reason"] for row in rows).items())),
        "thinking_close_tag": sum("</think>" in row["response"] for row in rows),
    }


def transition(old: dict[int, dict[str, Any]], new: dict[int, dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for index in range(EXPECTED_ROWS):
        old_correct = bool(old[index]["correct"])
        new_correct = bool(new[index]["correct"])
        if old_correct and new_correct:
            counts["both_correct"] += 1
        elif old_correct:
            counts["old_only"] += 1
        elif new_correct:
            counts["new_only"] += 1
        else:
            counts["both_wrong"] += 1
    return {key: counts[key] for key in ("both_correct", "old_only", "new_only", "both_wrong")}


def diagnostic_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"num_tokens": int(summary["token_diagnostics"]["num_tokens"])}
    for weighting in ("rollout_weighted", "token_weighted"):
        source = summary["token_diagnostics"][weighting]
        result[weighting] = {
            name: source[name]["mean"]
            for name in (
                "sampled_reverse_kl_k1",
                "sampled_reverse_kl_k3",
                "top1_agreement",
                "topk_overlap",
            )
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    args = parser.parse_args()

    new_rows: dict[str, dict[int, dict[str, Any]]] = {}
    old_rows: dict[str, dict[int, dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    output: dict[str, Any] = {
        "protocol": {
            "expected_rows_per_model": EXPECTED_ROWS,
            "max_response_tokens": EXPECTED_MAX_RESPONSE,
            "max_model_len": EXPECTED_MAX_MODEL_LEN,
        },
        "models": {},
        "diagnostics": {},
        "correctness_overlap": {},
    }

    for model, (merged_dir, filename) in MODELS.items():
        new_path = args.new_root / merged_dir / filename
        old_path = args.old_root / merged_dir / filename
        new_list = read_jsonl(new_path)
        old_list = read_jsonl(old_path)
        new_rows[model] = keyed(new_list, new_path)
        old_rows[model] = keyed(old_list, old_path)
        for index in range(EXPECTED_ROWS):
            for field in KEY_FIELDS:
                if new_rows[model][index][field] != old_rows[model][index][field]:
                    raise AssertionError(f"{model} index={index}: old/new mismatch in {field}")
        output["models"][model] = {
            "new": model_metrics(new_list),
            "old": model_metrics(old_list),
            "old_to_new": transition(old_rows[model], new_rows[model]),
        }

    reference = new_rows["qwen3_14b"]
    for model, rows in new_rows.items():
        for index in range(EXPECTED_ROWS):
            for field in KEY_FIELDS:
                if rows[index][field] != reference[index][field]:
                    raise AssertionError(f"{model} index={index}: cross-model mismatch in {field}")

    model_order = ("qwen3_1_7b", "qwen3_4b", "qwen3_14b")
    partition = Counter(
        "".join("1" if new_rows[model][index]["correct"] else "0" for model in model_order)
        for index in range(EXPECTED_ROWS)
    )
    pairwise: dict[str, Any] = {}
    for left_index, left in enumerate(model_order):
        for right in model_order[left_index + 1 :]:
            left_correct = {i for i, row in new_rows[left].items() if row["correct"]}
            right_correct = {i for i, row in new_rows[right].items() if row["correct"]}
            intersection = len(left_correct & right_correct)
            union = len(left_correct | right_correct)
            pairwise[f"{left}__{right}"] = {
                "correct_intersection": intersection,
                "correct_union": union,
                "correct_jaccard": intersection / union,
            }
    output["correctness_overlap"] = {
        "model_order": model_order,
        "three_way_partition": {bits: partition[bits] for bits in sorted(partition)},
        "pairwise": pairwise,
    }

    teacher_config = read_json(args.new_root / "qwen3_14b" / "run_config.json")
    if teacher_config["sampling"]["max_tokens"] != EXPECTED_MAX_RESPONSE:
        raise AssertionError("14B max_tokens mismatch")
    if teacher_config["max_model_len"] != EXPECTED_MAX_MODEL_LEN:
        raise AssertionError("14B max_model_len mismatch")

    for model, merged_dir in (("qwen3_4b", "merged"), ("qwen3_1_7b", "merged_qwen3_1_7b")):
        summary_path = args.new_root / merged_dir / "summary.json"
        summary = read_json(summary_path)
        summaries[model] = summary
        assert_finite_tree(summary, str(summary_path))
        token_path = args.new_root / merged_dir / "tokens.parquet"
        rollout_path = args.new_root / merged_dir / "per_rollout_diagnostics.parquet"
        token_rows = pq.ParquetFile(token_path).metadata.num_rows
        rollout_rows = pq.ParquetFile(rollout_path).metadata.num_rows
        expected_tokens = int(summary["token_diagnostics"]["num_tokens"])
        if token_rows != expected_tokens:
            raise AssertionError(f"{token_path}: {token_rows} != {expected_tokens}")
        if rollout_rows != EXPECTED_ROWS:
            raise AssertionError(f"{rollout_path}: expected 1000, got {rollout_rows}")
        sampling = summary["protocol"]["sampling"]
        if sampling["max_response_tokens"] != EXPECTED_MAX_RESPONSE:
            raise AssertionError(f"{model}: summary max_response_tokens mismatch")
        if sampling["max_model_len"] != EXPECTED_MAX_MODEL_LEN:
            raise AssertionError(f"{model}: summary max_model_len mismatch")
        output["diagnostics"][model] = diagnostic_metrics(summary)

    # The same merged teacher rollout must be consumed by both student analyses.
    teacher_a = (args.new_root / "merged" / "qwen3_14b_rollouts.jsonl").read_bytes()
    teacher_b = (args.new_root / "merged_qwen3_1_7b" / "qwen3_14b_rollouts.jsonl").read_bytes()
    if teacher_a != teacher_b:
        raise AssertionError("The two student analyses did not use the same 14B rollout file")

    print("DEEPMATH_MAXLEN_VERIFY_OK")
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
