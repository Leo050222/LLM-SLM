#!/usr/bin/env python3
"""Read-only aggregate analysis for direct-prefix and after-bridge rollouts."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train")
STUDENT = ROOT / "student_from_anchor_prefix_qwen3_1_7b_n8_seed20260827" / "analysis_correctness_length_single_job" / "scores.jsonl"
TEACHER = ROOT / "teacher_full_qwen3_14b_n8_seed20260826" / "analysis_correctness_length_single_job" / "scores.jsonl"
BRIDGE_ROOTS = [
    ROOT / "student_after_bridge_qwen3_1_7b_h256_n8_seed20260826",
] + [
    ROOT / "student_after_bridge_qwen3_1_7b_h256_n8_x8_seed20260826" / f"replicate_{index:02d}"
    for index in range(1, 8)
]


def rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def mean(values):
    return sum(values) / len(values)


def quantile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def pearson(xs, ys):
    mx, my = mean(xs), mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return numerator / denominator if denominator else None


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + end - 1) / 2 + 1
        for position in range(start, end):
            result[order[position]] = average_rank
        start = end
    return result


def correlations(xs, ys):
    return {
        "n": len(xs),
        "pearson_r": pearson(xs, ys),
        "spearman_rho": pearson(ranks(xs), ranks(ys)),
    }


def reduction_summary(values):
    reduced = [value for value in values if value > 0]
    equal = sum(value == 0 for value in values)
    return {
        "total": len(values),
        "reduced_count": len(reduced),
        "reduced_fraction": len(reduced) / len(values),
        "equal_count": equal,
        "increased_count": len(values) - len(reduced) - equal,
        "reduction_when_reduced_mean": mean(reduced),
        "reduction_when_reduced_median": statistics.median(reduced),
        "reduction_when_reduced_p25": quantile(reduced, 0.25),
        "reduction_when_reduced_p75": quantile(reduced, 0.75),
        "signed_reduction_all_mean": mean(values),
    }


def load_direct(path: Path):
    by_prefix = defaultdict(lambda: {"n": 0, "completion": 0, "full": 0, "correct": 0})
    totals = {"n": 0, "completion": 0, "full": 0, "correct": 0}
    for row in rows(path):
        index = int(row["dataset_index"])
        completion_value = row.get("completion_token_count")
        if completion_value is None:
            completion_value = row.get("student_completion_token_count")
        if completion_value is None:
            completion_value = row.get("teacher_completion_token_count")
        if completion_value is None:
            raise KeyError(f"no completion length in {path}: keys={sorted(row)}")
        completion = int(completion_value)
        full = int(row["full_trace_token_count"])
        correct = int(bool(row["correct"]))
        for target in (by_prefix[index], totals):
            target["n"] += 1
            target["completion"] += completion
            target["full"] += full
            target["correct"] += correct
    return by_prefix, totals


student_by_prefix, student_total = load_direct(STUDENT)
_, teacher_total = load_direct(TEACHER)

bridge_by_key = defaultdict(lambda: {"n": 0, "continuation": 0, "full": 0, "bridge": 0, "correct": 0})
bridge_total = {"n": 0, "continuation": 0, "full": 0, "bridge": 0, "correct": 0}
for root in BRIDGE_ROOTS:
    path = root / "analysis_correctness_length_single_job" / "scores.jsonl"
    for row in rows(path):
        key = (int(row["dataset_index"]), int(row["rollout_index"]))
        continuation = int(row["student_continuation_token_count"])
        full = int(row["full_trace_token_count"])
        bridge = int(row["source_bridge_token_count"])
        correct = int(bool(row["correct"]))
        for target in (bridge_by_key[key], bridge_total):
            target["n"] += 1
            target["continuation"] += continuation
            target["full"] += full
            target["bridge"] += bridge
            target["correct"] += correct

bridge_cont_reductions = []
bridge_full_reductions = []
bridge_after_accuracy = []
bridge_accuracy_lifts = []
good_records = []
good_prefixes = set()
good_count_by_prefix = Counter()
lift_step_histogram = Counter()
baseline_correct_histogram_good = Counter()
after_correct_histogram_good = Counter()
prefix_bridge = defaultdict(lambda: {"n": 0, "continuation": 0, "full": 0, "correct": 0})
for (dataset_index, _), value in bridge_by_key.items():
    baseline = student_by_prefix[dataset_index]
    base_cont = baseline["completion"] / baseline["n"]
    base_full = baseline["full"] / baseline["n"]
    base_acc = baseline["correct"] / baseline["n"]
    after_cont = value["continuation"] / value["n"]
    after_full = value["full"] / value["n"]
    after_acc = value["correct"] / value["n"]
    bridge_cont_reductions.append(base_cont - after_cont)
    bridge_full_reductions.append(base_full - after_full)
    bridge_after_accuracy.append(after_acc)
    bridge_accuracy_lifts.append(after_acc - base_acc)
    if value["correct"] > baseline["correct"]:
        lift_steps = value["correct"] - baseline["correct"]
        good_prefixes.add(dataset_index)
        good_count_by_prefix[dataset_index] += 1
        lift_step_histogram[lift_steps] += 1
        baseline_correct_histogram_good[baseline["correct"]] += 1
        after_correct_histogram_good[value["correct"]] += 1
        good_records.append({
            "dataset_index": dataset_index,
            "base_accuracy": base_acc,
            "after_accuracy": after_acc,
            "accuracy_lift": after_acc - base_acc,
            "base_continuation": base_cont,
            "after_continuation": after_cont,
            "continuation_reduction": base_cont - after_cont,
            "base_full": base_full,
            "after_full": after_full,
            "full_reduction": base_full - after_full,
        })
    aggregate = prefix_bridge[dataset_index]
    aggregate["n"] += value["n"]
    aggregate["continuation"] += value["continuation"]
    aggregate["full"] += value["full"]
    aggregate["correct"] += value["correct"]

prefix_cont_reductions = []
prefix_full_reductions = []
prefix_after_accuracy = []
prefix_accuracy_lifts = []
for dataset_index, value in prefix_bridge.items():
    baseline = student_by_prefix[dataset_index]
    base_cont = baseline["completion"] / baseline["n"]
    base_full = baseline["full"] / baseline["n"]
    base_acc = baseline["correct"] / baseline["n"]
    after_cont = value["continuation"] / value["n"]
    after_full = value["full"] / value["n"]
    after_acc = value["correct"] / value["n"]
    prefix_cont_reductions.append(base_cont - after_cont)
    prefix_full_reductions.append(base_full - after_full)
    prefix_after_accuracy.append(after_acc)
    prefix_accuracy_lifts.append(after_acc - base_acc)


def subset_correlations(reductions, outcomes):
    selected = [(x, y) for x, y in zip(reductions, outcomes) if x > 0]
    return correlations([pair[0] for pair in selected], [pair[1] for pair in selected])


def outcome_by_reduction_group(reductions, after_accuracy, accuracy_lifts):
    output = {}
    for label, predicate in (
        ("reduced", lambda value: value > 0),
        ("not_reduced", lambda value: value <= 0),
    ):
        selected = [
            (after, lift)
            for reduction, after, lift in zip(reductions, after_accuracy, accuracy_lifts)
            if predicate(reduction)
        ]
        output[label] = {
            "count": len(selected),
            "mean_after_accuracy": mean([item[0] for item in selected]),
            "mean_accuracy_lift": mean([item[1] for item in selected]),
        }
    return output


result = {
    "definitions": {
        "continuation_length": "tokens newly generated by the evaluated model after its conditioning context",
        "full_trace_length": "prefix + optional 256-token bridge + evaluated-model continuation",
        "signed_reduction": "direct Student mean length minus after-bridge mean length; positive means shorter after bridge",
        "bridge_level_pairing": "each of 105,952 bridges (8 after-bridge replicates) compared with the 8 direct Student rollouts for the same prefix",
        "prefix_level_pairing": "all 64 after-bridge continuations under a prefix compared with its 8 direct Student rollouts",
    },
    "average_lengths": {
        "student_from_prefix": {
            "rollouts": student_total["n"],
            "mean_continuation_tokens": student_total["completion"] / student_total["n"],
            "mean_full_trace_tokens": student_total["full"] / student_total["n"],
        },
        "student_after_bridge": {
            "rollouts": bridge_total["n"],
            "mean_continuation_tokens": bridge_total["continuation"] / bridge_total["n"],
            "mean_bridge_tokens": bridge_total["bridge"] / bridge_total["n"],
            "mean_full_trace_tokens": bridge_total["full"] / bridge_total["n"],
        },
        "teacher_from_prefix": {
            "rollouts": teacher_total["n"],
            "mean_continuation_tokens": teacher_total["completion"] / teacher_total["n"],
            "mean_full_trace_tokens": teacher_total["full"] / teacher_total["n"],
        },
    },
    "good_bridge_analysis": {
        "definition": "after-bridge correct count among 8 is strictly greater than direct-Student correct count among 8 for the same prefix",
        "good_bridge_count": len(good_records),
        "total_bridge_count": len(bridge_by_key),
        "good_bridge_fraction": len(good_records) / len(bridge_by_key),
        "prefixes_with_at_least_one_good_bridge": len(good_prefixes),
        "prefix_fraction_with_at_least_one_good_bridge": len(good_prefixes) / len(student_by_prefix),
        "good_bridges_per_prefix_histogram": {
            str(count): sum(1 for index in student_by_prefix if good_count_by_prefix[index] == count)
            for count in range(9)
        },
        "accuracy": {
            "mean_direct_student_accuracy": mean([row["base_accuracy"] for row in good_records]),
            "mean_after_bridge_accuracy": mean([row["after_accuracy"] for row in good_records]),
            "mean_accuracy_lift": mean([row["accuracy_lift"] for row in good_records]),
            "median_accuracy_lift": statistics.median([row["accuracy_lift"] for row in good_records]),
            "lift_steps_out_of_8_histogram": {str(step): lift_step_histogram[step] for step in range(1, 9)},
            "baseline_correct_count_histogram": {str(count): baseline_correct_histogram_good[count] for count in range(9)},
            "after_correct_count_histogram": {str(count): after_correct_histogram_good[count] for count in range(9)},
        },
        "continuation_length": {
            "mean_direct_tokens": mean([row["base_continuation"] for row in good_records]),
            "mean_after_bridge_tokens": mean([row["after_continuation"] for row in good_records]),
            "mean_signed_reduction": mean([row["continuation_reduction"] for row in good_records]),
            "median_signed_reduction": statistics.median([row["continuation_reduction"] for row in good_records]),
            "reduced_count": sum(row["continuation_reduction"] > 0 for row in good_records),
            "reduced_fraction": mean([row["continuation_reduction"] > 0 for row in good_records]),
        },
        "full_trace_length": {
            "mean_direct_tokens": mean([row["base_full"] for row in good_records]),
            "mean_after_bridge_tokens": mean([row["after_full"] for row in good_records]),
            "mean_signed_reduction": mean([row["full_reduction"] for row in good_records]),
            "median_signed_reduction": statistics.median([row["full_reduction"] for row in good_records]),
            "reduced_count": sum(row["full_reduction"] > 0 for row in good_records),
            "reduced_fraction": mean([row["full_reduction"] > 0 for row in good_records]),
        },
    },
    "bridge_level": {
        "continuation_length": reduction_summary(bridge_cont_reductions),
        "full_trace_length": reduction_summary(bridge_full_reductions),
        "correlation_reduction_vs_after_accuracy_all": correlations(bridge_cont_reductions, bridge_after_accuracy),
        "correlation_reduction_vs_accuracy_lift_all": correlations(bridge_cont_reductions, bridge_accuracy_lifts),
        "correlation_reduction_vs_after_accuracy_reduced_only": subset_correlations(bridge_cont_reductions, bridge_after_accuracy),
        "correlation_reduction_vs_accuracy_lift_reduced_only": subset_correlations(bridge_cont_reductions, bridge_accuracy_lifts),
        "full_trace_correlation_reduction_vs_after_accuracy_reduced_only": subset_correlations(bridge_full_reductions, bridge_after_accuracy),
        "full_trace_correlation_reduction_vs_accuracy_lift_reduced_only": subset_correlations(bridge_full_reductions, bridge_accuracy_lifts),
        "continuation_outcome_by_reduction_group": outcome_by_reduction_group(bridge_cont_reductions, bridge_after_accuracy, bridge_accuracy_lifts),
        "full_trace_outcome_by_reduction_group": outcome_by_reduction_group(bridge_full_reductions, bridge_after_accuracy, bridge_accuracy_lifts),
    },
    "prefix_level": {
        "continuation_length": reduction_summary(prefix_cont_reductions),
        "full_trace_length": reduction_summary(prefix_full_reductions),
        "correlation_reduction_vs_after_accuracy_all": correlations(prefix_cont_reductions, prefix_after_accuracy),
        "correlation_reduction_vs_accuracy_lift_all": correlations(prefix_cont_reductions, prefix_accuracy_lifts),
        "correlation_reduction_vs_after_accuracy_reduced_only": subset_correlations(prefix_cont_reductions, prefix_after_accuracy),
        "correlation_reduction_vs_accuracy_lift_reduced_only": subset_correlations(prefix_cont_reductions, prefix_accuracy_lifts),
        "full_trace_correlation_reduction_vs_after_accuracy_reduced_only": subset_correlations(prefix_full_reductions, prefix_after_accuracy),
        "full_trace_correlation_reduction_vs_accuracy_lift_reduced_only": subset_correlations(prefix_full_reductions, prefix_accuracy_lifts),
        "continuation_outcome_by_reduction_group": outcome_by_reduction_group(prefix_cont_reductions, prefix_after_accuracy, prefix_accuracy_lifts),
        "full_trace_outcome_by_reduction_group": outcome_by_reduction_group(prefix_full_reductions, prefix_after_accuracy, prefix_accuracy_lifts),
    },
}
print(json.dumps(result, indent=2, sort_keys=True))
