#!/usr/bin/env python3
"""Aggregate whole-trace and continuation lengths for prefix rollout experiments.

This script is intended to run on the H800 login node.  It reads only the
merged correctness/length score files and emits a compact JSON report.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train")
SCORE_SUFFIX = Path("analysis_correctness_length_single_job/scores.jsonl")
EXPERIMENTS = {
    "student_from_prefix": ROOT
    / "student_from_anchor_prefix_qwen3_1_7b_n8_seed20260827"
    / SCORE_SUFFIX,
    "teacher_from_prefix": ROOT
    / "teacher_full_qwen3_14b_n8_seed20260826"
    / SCORE_SUFFIX,
    "student_after_bridge_rep0": ROOT
    / "student_after_bridge_qwen3_1_7b_h256_n8_seed20260826"
    / SCORE_SUFFIX,
}
for replicate in range(1, 8):
    EXPERIMENTS[f"student_after_bridge_rep{replicate}"] = (
        ROOT
        / "student_after_bridge_qwen3_1_7b_h256_n8_x8_seed20260826"
        / f"replicate_{replicate:02d}"
        / SCORE_SUFFIX
    )


def quantile(sorted_values: list[float], q: float) -> float:
    position = (len(sorted_values) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    return (
        sorted_values[low] * (high - position)
        + sorted_values[high] * (position - low)
    )


def describe(values: list[int] | list[float]) -> dict:
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "total": sum(ordered),
        "mean": statistics.fmean(ordered),
        "std_population": statistics.pstdev(ordered),
        "min": ordered[0],
        "p10": quantile(ordered, 0.10),
        "p25": quantile(ordered, 0.25),
        "median": quantile(ordered, 0.50),
        "p75": quantile(ordered, 0.75),
        "p90": quantile(ordered, 0.90),
        "p95": quantile(ordered, 0.95),
        "p99": quantile(ordered, 0.99),
        "max": ordered[-1],
    }


def load_experiment(name: str, path: Path) -> tuple[dict, dict[int, list[int]]]:
    full: list[int] = []
    continuation: list[int] = []
    prefix: list[int] = []
    bridge: list[int] = []
    correct_full: list[int] = []
    wrong_full: list[int] = []
    finishes: Counter[str] = Counter()
    dataset_indices: set[int] = set()
    rollout_indices: set[int] = set()
    full_by_prefix: dict[int, list[int]] = defaultdict(list)

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            full_value = int(row["full_trace_token_count"])
            continuation_value = row.get("student_continuation_token_count")
            if continuation_value is None:
                continuation_value = row.get("student_completion_token_count")
            if continuation_value is None:
                continuation_value = row.get("teacher_completion_token_count")
            if continuation_value is None:
                continuation_value = row["completion_token_count"]
            prefix_value = int(row["response_prefix_token_count"])
            dataset_index = int(row["dataset_index"])

            full.append(full_value)
            continuation.append(int(continuation_value))
            prefix.append(prefix_value)
            full_by_prefix[dataset_index].append(full_value)
            dataset_indices.add(dataset_index)
            rollout_indices.add(int(row["rollout_index"]))
            finishes[str(row.get("finish_reason", "missing"))] += 1
            (correct_full if bool(row["correct"]) else wrong_full).append(full_value)
            if "source_bridge_token_count" in row:
                bridge.append(int(row["source_bridge_token_count"]))

    per_prefix_means = [statistics.fmean(values) for values in full_by_prefix.values()]
    summary = {
        "source": str(path),
        "rollouts": len(full),
        "unique_prefixes": len(dataset_indices),
        "rollout_indices": sorted(rollout_indices),
        "finish_reasons": dict(sorted(finishes.items())),
        "correct_rollouts": len(correct_full),
        "accuracy": len(correct_full) / len(full),
        "prefix_tokens": describe(prefix),
        "bridge_tokens": describe(bridge) if bridge else None,
        "continuation_tokens": describe(continuation),
        "whole_trace_tokens": describe(full),
        "whole_trace_tokens_correct": describe(correct_full),
        "whole_trace_tokens_wrong": describe(wrong_full),
        "per_prefix_mean_whole_trace_tokens": describe(per_prefix_means),
        "whole_trace_threshold_counts": {
            str(threshold): sum(value >= threshold for value in full)
            for threshold in (2048, 4096, 8192, 16384, 32768, 40960)
        },
    }
    return summary, full_by_prefix


summaries: dict[str, dict] = {}
by_prefix: dict[str, dict[int, list[int]]] = {}
for experiment_name, experiment_path in EXPERIMENTS.items():
    summaries[experiment_name], by_prefix[experiment_name] = load_experiment(
        experiment_name, experiment_path
    )

bridge_names = [f"student_after_bridge_rep{index}" for index in range(8)]
bridge_rep_means = [
    summaries[name]["whole_trace_tokens"]["mean"] for name in bridge_names
]
bridge_rep_medians = [
    summaries[name]["whole_trace_tokens"]["median"] for name in bridge_names
]

baseline_prefix_means = {
    index: statistics.fmean(values)
    for index, values in by_prefix["student_from_prefix"].items()
}

def paired_prefix_delta(experiment_name: str) -> dict:
    deltas = []
    for index, values in by_prefix[experiment_name].items():
        deltas.append(statistics.fmean(values) - baseline_prefix_means[index])
    result = describe(deltas)
    result.update(
        {
            "shorter_prefix_count": sum(value < 0 for value in deltas),
            "equal_prefix_count": sum(value == 0 for value in deltas),
            "longer_prefix_count": sum(value > 0 for value in deltas),
        }
    )
    return result


all_bridge_full = []
all_bridge_continuation = []
all_bridge_correct_full = []
all_bridge_wrong_full = []
all_bridge_finishes: Counter[str] = Counter()
for name in bridge_names:
    path = EXPERIMENTS[name]
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            full_value = int(row["full_trace_token_count"])
            cont_value = int(row["student_continuation_token_count"])
            all_bridge_full.append(full_value)
            all_bridge_continuation.append(cont_value)
            (all_bridge_correct_full if bool(row["correct"]) else all_bridge_wrong_full).append(full_value)
            all_bridge_finishes[str(row.get("finish_reason", "missing"))] += 1

baseline_mean = summaries["student_from_prefix"]["whole_trace_tokens"]["mean"]
teacher_mean = summaries["teacher_from_prefix"]["whole_trace_tokens"]["mean"]
bridge_mean = statistics.fmean(all_bridge_full)

report = {
    "definitions": {
        "whole_trace_tokens": "response_prefix_token_count + optional source_bridge_token_count + generated continuation token count",
        "continuation_tokens": "tokens generated after the experiment-specific conditioning context",
        "population_std": "standard deviation over all rollouts, ddof=0",
        "formal_only": "smoke and retry directories are excluded",
    },
    "experiments": summaries,
    "student_after_bridge_all_replicates": {
        "rollouts": len(all_bridge_full),
        "unique_prefixes": len(baseline_prefix_means),
        "replicates": 8,
        "finish_reasons": dict(sorted(all_bridge_finishes.items())),
        "accuracy": len(all_bridge_correct_full) / len(all_bridge_full),
        "continuation_tokens": describe(all_bridge_continuation),
        "whole_trace_tokens": describe(all_bridge_full),
        "whole_trace_tokens_correct": describe(all_bridge_correct_full),
        "whole_trace_tokens_wrong": describe(all_bridge_wrong_full),
        "replicate_mean_whole_trace_tokens": describe(bridge_rep_means),
        "replicate_median_whole_trace_tokens": describe(bridge_rep_medians),
    },
    "comparisons_vs_student_from_prefix": {
        "teacher_mean_delta_tokens": teacher_mean - baseline_mean,
        "teacher_mean_relative_delta": teacher_mean / baseline_mean - 1,
        "bridge_all_mean_delta_tokens": bridge_mean - baseline_mean,
        "bridge_all_mean_relative_delta": bridge_mean / baseline_mean - 1,
        "teacher_paired_per_prefix_delta": paired_prefix_delta("teacher_from_prefix"),
        "bridge_paired_per_prefix_delta_by_replicate": {
            name: paired_prefix_delta(name) for name in bridge_names
        },
    },
}

print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
