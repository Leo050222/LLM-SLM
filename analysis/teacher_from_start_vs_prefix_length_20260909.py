#!/usr/bin/env python3
"""Compare Teacher rollout lengths from problem start versus from an anchor prefix."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path("/storage/yukaichengLab/wangxin/test_eiloeo")
FROM_START = (
    ROOT
    / "opd_rollouts/deepmath_ge6_train25pct_qwen3_14b_native_nonthinking_seed20260811_single_node_8h800"
    / "scoring_rule_v1/scores.jsonl"
)
FROM_PREFIX = (
    ROOT
    / "dataset/train/teacher_full_qwen3_14b_n8_seed20260826"
    / "analysis_correctness_length_single_job/scores.jsonl"
)


def describe(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


start = {}
with FROM_START.open("r", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("padded_duplicate", False):
            continue
        problem_id = str(row["problem_id"])
        if problem_id in start:
            raise RuntimeError(f"Duplicate from-start problem: {problem_id}")
        start[problem_id] = {
            "tokens": int(row["response_tokens"]),
            "correct": int(bool(row["correct"])),
            "has_length": row.get("finish_reason") == "length",
        }

prefix = defaultdict(lambda: {"n": 0, "prefix": 0, "cont": 0, "full": 0, "correct": 0, "has_length": False})
with FROM_PREFIX.open("r", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        target = prefix[str(row["problem_id"])]
        target["n"] += 1
        target["prefix"] += int(row["response_prefix_token_count"])
        target["cont"] += int(row["teacher_completion_token_count"])
        target["full"] += int(row["full_trace_token_count"])
        target["correct"] += int(bool(row["correct"]))
        target["has_length"] |= row.get("finish_reason") == "length"

if any(value["n"] != 8 for value in prefix.values()):
    raise RuntimeError("Expected exactly 8 Teacher-from-prefix rollouts per problem")
ids = sorted(prefix)
if len(ids) != 13244 or not set(ids).issubset(start):
    raise RuntimeError("Unexpected problem coverage")

start_tokens = np.asarray([start[i]["tokens"] for i in ids], dtype=float)
prefix_tokens = np.asarray([prefix[i]["prefix"] / 8 for i in ids], dtype=float)
prefix_cont = np.asarray([prefix[i]["cont"] / 8 for i in ids], dtype=float)
prefix_full = np.asarray([prefix[i]["full"] / 8 for i in ids], dtype=float)
clean = np.asarray([not (start[i]["has_length"] or prefix[i]["has_length"]) for i in ids])


def paired_summary(mask: np.ndarray) -> dict:
    direct = start_tokens[mask]
    conditioned_cont = prefix_cont[mask]
    conditioned_full = prefix_full[mask]
    conditioned_prefix = prefix_tokens[mask]
    continuation_delta = conditioned_cont - direct
    full_delta = conditioned_full - direct
    sem = stats.sem(full_delta)
    ci = stats.t.interval(0.95, len(full_delta) - 1, loc=full_delta.mean(), scale=sem)
    paired_t = stats.ttest_1samp(full_delta, 0.0)
    wilcoxon = stats.wilcoxon(full_delta)
    return {
        "problems": int(mask.sum()),
        "from_start_generated_tokens": describe(direct),
        "from_prefix_existing_prefix_tokens": describe(conditioned_prefix),
        "from_prefix_generated_continuation_tokens": describe(conditioned_cont),
        "from_prefix_full_trace_tokens": describe(conditioned_full),
        "from_prefix_minus_start_generated_delta": describe(continuation_delta),
        "from_prefix_full_minus_start_delta": describe(full_delta),
        "full_trace_direction": {
            "from_prefix_shorter_count": int((full_delta < 0).sum()),
            "equal_count": int((full_delta == 0).sum()),
            "from_prefix_longer_count": int((full_delta > 0).sum()),
            "from_prefix_shorter_fraction": float((full_delta < 0).mean()),
            "from_prefix_longer_fraction": float((full_delta > 0).mean()),
        },
        "mean_full_delta_95pct_ci": [float(ci[0]), float(ci[1])],
        "paired_t": {"statistic": float(paired_t.statistic), "p": float(paired_t.pvalue)},
        "wilcoxon": {"statistic": float(wilcoxon.statistic), "p": float(wilcoxon.pvalue)},
    }


report = {
    "definitions": {
        "from_start": "Teacher response generated from the full problem, one rollout per problem",
        "from_prefix_continuation": "Teacher tokens newly generated after the saved anchor prefix, eight rollouts per problem",
        "from_prefix_full_trace": "saved anchor prefix plus Teacher continuation",
        "delta": "from-prefix value minus from-start value; negative means from-prefix is shorter",
        "comparison_caveat": "paired by problem_id, but from-start has one stochastic rollout while from-prefix is an eight-rollout mean",
    },
    "all": paired_summary(np.ones(len(ids), dtype=bool)),
    "clean": paired_summary(clean),
}

print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
