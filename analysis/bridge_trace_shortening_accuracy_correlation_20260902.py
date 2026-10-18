#!/usr/bin/env python3
"""Correlation between bridge-induced trace shortening and accuracy lift."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train")
SUFFIX = Path("analysis_correctness_length_single_job/scores.jsonl")
DIRECT = ROOT / "student_from_anchor_prefix_qwen3_1_7b_n8_seed20260827" / SUFFIX
BRIDGES = [ROOT / "student_after_bridge_qwen3_1_7b_h256_n8_seed20260826" / SUFFIX]
BRIDGES.extend(
    ROOT
    / "student_after_bridge_qwen3_1_7b_h256_n8_x8_seed20260826"
    / f"replicate_{i:02d}"
    / SUFFIX
    for i in range(1, 8)
)


def read_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def summary(records: list[dict]) -> dict:
    x = np.asarray([row["full_reduction"] for row in records], dtype=float)
    xc = np.asarray([row["continuation_reduction"] for row in records], dtype=float)
    y = np.asarray([row["accuracy_lift"] for row in records], dtype=float)
    after_accuracy = np.asarray([row["after_accuracy"] for row in records], dtype=float)
    base_accuracy = np.asarray([row["base_accuracy"] for row in records], dtype=float)
    base_full = np.asarray([row["base_full"] for row in records], dtype=float)

    design = np.column_stack([np.ones(len(records)), base_accuracy, base_full])
    residual_x = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    residual_y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]

    shortened = x > 0
    improved = y > 0
    equal_accuracy = y == 0
    slope = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x) * 100.0)

    quantile_edges = np.quantile(x, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    bins = []
    for index in range(5):
        if index == 4:
            mask = (x >= quantile_edges[index]) & (x <= quantile_edges[index + 1])
        else:
            mask = (x >= quantile_edges[index]) & (x < quantile_edges[index + 1])
        bins.append(
            {
                "n": int(mask.sum()),
                "reduction_min": float(x[mask].min()),
                "reduction_max": float(x[mask].max()),
                "mean_reduction_tokens": float(x[mask].mean()),
                "mean_accuracy_lift": float(y[mask].mean()),
                "mean_after_accuracy": float(after_accuracy[mask].mean()),
                "improved_fraction": float(improved[mask].mean()),
            }
        )

    by_shortening = {}
    for label, mask in (("shorter", shortened), ("not_shorter", ~shortened)):
        by_shortening[label] = {
            "n": int(mask.sum()),
            "fraction": float(mask.mean()),
            "mean_reduction_tokens": float(x[mask].mean()),
            "mean_accuracy_lift": float(y[mask].mean()),
            "mean_after_accuracy": float(after_accuracy[mask].mean()),
            "improved_fraction": float(improved[mask].mean()),
        }

    return {
        "n": len(records),
        "mean_full_reduction_tokens": float(x.mean()),
        "median_full_reduction_tokens": float(np.median(x)),
        "mean_accuracy_lift": float(y.mean()),
        "pearson_full_reduction_vs_accuracy_lift": corr(x, y),
        "spearman_full_reduction_vs_accuracy_lift": corr(rankdata(x), rankdata(y)),
        "pearson_continuation_reduction_vs_accuracy_lift": corr(xc, y),
        "spearman_continuation_reduction_vs_accuracy_lift": corr(rankdata(xc), rankdata(y)),
        "pearson_full_reduction_vs_after_accuracy": corr(x, after_accuracy),
        "partial_pearson_controlling_baseline_accuracy_and_length": corr(residual_x, residual_y),
        "ols_accuracy_lift_per_100_shorter_tokens": slope,
        "shorter_fraction": float(shortened.mean()),
        "improved_fraction": float(improved.mean()),
        "equal_accuracy_fraction": float(equal_accuracy.mean()),
        "joint_fractions": {
            "shorter_and_improved": float((shortened & improved).mean()),
            "shorter_and_not_improved": float((shortened & ~improved).mean()),
            "not_shorter_and_improved": float((~shortened & improved).mean()),
            "not_shorter_and_not_improved": float((~shortened & ~improved).mean()),
        },
        "by_shortening_group": by_shortening,
        "reduction_quintiles": bins,
    }


direct = defaultdict(lambda: {"n": 0, "full": 0, "cont": 0, "correct": 0, "has_length": False})
for row in read_rows(DIRECT):
    target = direct[int(row["dataset_index"])]
    target["n"] += 1
    target["full"] += int(row["full_trace_token_count"])
    target["cont"] += int(row["student_completion_token_count"])
    target["correct"] += int(bool(row["correct"]))
    target["has_length"] |= row.get("finish_reason") == "length"

bridge = defaultdict(lambda: {"n": 0, "full": 0, "cont": 0, "correct": 0, "has_length": False})
for path in BRIDGES:
    for row in read_rows(path):
        key = (int(row["dataset_index"]), int(row["rollout_index"]))
        target = bridge[key]
        target["n"] += 1
        target["full"] += int(row["full_trace_token_count"])
        target["cont"] += int(row["student_continuation_token_count"])
        target["correct"] += int(bool(row["correct"]))
        target["has_length"] |= row.get("finish_reason") == "length"


bridge_records = []
prefix_accumulator = defaultdict(lambda: {"n": 0, "full": 0, "cont": 0, "correct": 0, "has_length": False})
for (dataset_index, rollout_index), after in bridge.items():
    base = direct[dataset_index]
    base_full = base["full"] / base["n"]
    base_cont = base["cont"] / base["n"]
    base_acc = base["correct"] / base["n"]
    row = {
        "dataset_index": dataset_index,
        "rollout_index": rollout_index,
        "base_full": base_full,
        "base_accuracy": base_acc,
        "after_accuracy": after["correct"] / after["n"],
        "accuracy_lift": after["correct"] / after["n"] - base_acc,
        "full_reduction": base_full - after["full"] / after["n"],
        "continuation_reduction": base_cont - after["cont"] / after["n"],
        "has_length": base["has_length"] or after["has_length"],
    }
    bridge_records.append(row)
    aggregate = prefix_accumulator[dataset_index]
    for field in ("n", "full", "cont", "correct"):
        aggregate[field] += after[field]
    aggregate["has_length"] |= after["has_length"]


prefix_records = []
for dataset_index, after in prefix_accumulator.items():
    base = direct[dataset_index]
    base_full = base["full"] / base["n"]
    base_cont = base["cont"] / base["n"]
    base_acc = base["correct"] / base["n"]
    prefix_records.append(
        {
            "dataset_index": dataset_index,
            "base_full": base_full,
            "base_accuracy": base_acc,
            "after_accuracy": after["correct"] / after["n"],
            "accuracy_lift": after["correct"] / after["n"] - base_acc,
            "full_reduction": base_full - after["full"] / after["n"],
            "continuation_reduction": base_cont - after["cont"] / after["n"],
            "has_length": base["has_length"] or after["has_length"],
        }
    )


def baseline_strata(records: list[dict]) -> dict:
    result = {}
    for correct_count in range(9):
        selected = [row for row in records if round(row["base_accuracy"] * 8) == correct_count]
        if not selected:
            continue
        x = np.asarray([row["full_reduction"] for row in selected], dtype=float)
        y = np.asarray([row["accuracy_lift"] for row in selected], dtype=float)
        result[str(correct_count)] = {
            "n": len(selected),
            "pearson": corr(x, y),
            "spearman": corr(rankdata(x), rankdata(y)),
            "mean_reduction_tokens": float(x.mean()),
            "mean_accuracy_lift": float(y.mean()),
        }
    return result


report = {
    "definitions": {
        "full_reduction": "direct Student mean whole-trace tokens minus after-bridge mean whole-trace tokens; positive means bridge is shorter",
        "accuracy_lift": "after-bridge accuracy minus direct Student accuracy for the same prefix",
        "bridge_level": "105,952 source bridges; each after value averages its 8 replicate continuations and compares with 8 direct rollouts for that prefix",
        "prefix_level": "13,244 prefixes; each after value averages all 8 bridges x 8 replicates (64 continuations) and compares with 8 direct rollouts",
    },
    "bridge_level_all": summary(bridge_records),
    "bridge_level_without_any_length_finish": summary([row for row in bridge_records if not row["has_length"]]),
    "prefix_level_all": summary(prefix_records),
    "prefix_level_without_any_length_finish": summary([row for row in prefix_records if not row["has_length"]]),
    "prefix_level_by_direct_correct_count_out_of_8": baseline_strata(prefix_records),
}
print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
