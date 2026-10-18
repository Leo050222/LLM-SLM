#!/usr/bin/env python3
"""Strict read-only comparison of h32/h64/h128/h256 student-after-bridge results."""

from __future__ import annotations

import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path("/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train")
SCORE_SUFFIX = Path("analysis_correctness_length_single_job/scores.jsonl")
DIRECT = ROOT / "student_from_anchor_prefix_qwen3_1_7b_n8_seed20260827" / SCORE_SUFFIX
HORIZON_ROOTS = {
    32: ROOT / "student_after_bridge_qwen3_1_7b_h32_n8_x8_seed20260826",
    64: ROOT / "student_after_bridge_qwen3_1_7b_h64_n8_x8_seed20260826",
    128: ROOT / "student_after_bridge_qwen3_1_7b_h128_n8_x8_seed20260826",
}
H256_REP0 = ROOT / "student_after_bridge_qwen3_1_7b_h256_n8_seed20260826" / SCORE_SUFFIX
H256_X8 = ROOT / "student_after_bridge_qwen3_1_7b_h256_n8_x8_seed20260826"
EXPECTED_PREFIXES = 13_244
EXPECTED_ROLLOUTS = EXPECTED_PREFIXES * 8


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def describe(values) -> dict:
    x = np.asarray(values, dtype=float)
    return {
        "n": int(x.size),
        "total": float(x.sum()),
        "mean": float(x.mean()),
        "std_population": float(x.std()),
        "min": float(x.min()),
        "p25": float(np.quantile(x, 0.25)),
        "median": float(np.median(x)),
        "p75": float(np.quantile(x, 0.75)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(x.max()),
    }


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


def corr(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def binary_association(exposure, outcome) -> dict:
    """Summarize association between two prefix-level binary indicators."""
    exposure = np.asarray(exposure, dtype=bool)
    outcome = np.asarray(outcome, dtype=bool)
    n11 = int((exposure & outcome).sum())
    n10 = int((exposure & ~outcome).sum())
    n01 = int((~exposure & outcome).sum())
    n00 = int((~exposure & ~outcome).sum())
    p1 = n11 / (n11 + n10) if n11 + n10 else float("nan")
    p0 = n01 / (n01 + n00) if n01 + n00 else float("nan")
    return {
        "n11_both_true": n11,
        "n10_exposure_only": n10,
        "n01_outcome_only": n01,
        "n00_both_false": n00,
        "phi_correlation": corr(exposure.astype(float), outcome.astype(float)),
        "outcome_given_exposure_fraction": p1,
        "outcome_given_no_exposure_fraction": p0,
        "risk_difference": p1 - p0,
        "risk_ratio": p1 / p0 if p0 > 0 else float("inf"),
        "odds_ratio": (n11 * n00) / (n10 * n01) if n10 > 0 and n01 > 0 else float("inf"),
    }


def rep_score_files(horizon: int, replicate: int) -> list[Path]:
    if horizon == 256:
        merged = H256_REP0 if replicate == 0 else H256_X8 / f"replicate_{replicate:02d}" / SCORE_SUFFIX
        return [merged]
    analysis = HORIZON_ROOTS[horizon] / f"replicate_{replicate:02d}" / "analysis_correctness_length_single_job"
    merged = analysis / "scores.jsonl"
    if merged.exists():
        return [merged]
    shards = sorted((analysis / "shards").glob("shard_*/scores.jsonl"))
    if len(shards) != 64:
        raise RuntimeError(f"h{horizon} rep{replicate}: expected 64 score shards, got {len(shards)}")
    return shards


# Direct baseline arrays.
direct_rows = list(iter_jsonl(DIRECT))
if len(direct_rows) != EXPECTED_ROLLOUTS:
    raise RuntimeError(f"direct row count {len(direct_rows)} != {EXPECTED_ROLLOUTS}")
dataset_ids = sorted({int(row["dataset_index"]) for row in direct_rows})
if len(dataset_ids) != EXPECTED_PREFIXES:
    raise RuntimeError(f"direct prefix count {len(dataset_ids)} != {EXPECTED_PREFIXES}")
id_to_pos = {dataset_id: pos for pos, dataset_id in enumerate(dataset_ids)}

direct_full_sum = np.zeros(EXPECTED_PREFIXES, dtype=np.int64)
direct_cont_sum = np.zeros(EXPECTED_PREFIXES, dtype=np.int64)
direct_correct = np.zeros(EXPECTED_PREFIXES, dtype=np.int16)
direct_count = np.zeros(EXPECTED_PREFIXES, dtype=np.int16)
direct_has_length = np.zeros(EXPECTED_PREFIXES, dtype=bool)
direct_problem_ids = [None] * EXPECTED_PREFIXES
direct_full_values = []
direct_cont_values = []
direct_finish = Counter()
direct_keys = set()
for row in direct_rows:
    pos = id_to_pos[int(row["dataset_index"])]
    problem_id = str(row["problem_id"])
    if direct_problem_ids[pos] is not None and direct_problem_ids[pos] != problem_id:
        raise RuntimeError(f"inconsistent problem_id for dataset position {pos}")
    direct_problem_ids[pos] = problem_id
    key = (pos, int(row["rollout_index"]))
    if key in direct_keys:
        raise RuntimeError(f"duplicate direct key {key}")
    direct_keys.add(key)
    full = int(row["full_trace_token_count"])
    cont = int(row.get("student_completion_token_count", row["completion_token_count"]))
    direct_full_sum[pos] += full
    direct_cont_sum[pos] += cont
    direct_correct[pos] += int(bool(row["correct"]))
    direct_count[pos] += 1
    direct_has_length[pos] |= row.get("finish_reason") == "length"
    direct_full_values.append(full)
    direct_cont_values.append(cont)
    direct_finish[str(row.get("finish_reason", "missing"))] += 1
if not np.all(direct_count == 8):
    raise RuntimeError("direct coverage is not exactly 8 per prefix")

direct_full_mean = direct_full_sum / 8.0
direct_cont_mean = direct_cont_sum / 8.0
direct_acc = direct_correct / 8.0


def association_summary(reduction, lift, after_accuracy, clean_mask) -> dict:
    reduction = np.asarray(reduction, dtype=float)
    lift = np.asarray(lift, dtype=float)
    after_accuracy = np.asarray(after_accuracy, dtype=float)
    clean_mask = np.asarray(clean_mask, dtype=bool)
    output = {}
    for label, mask in (("all", np.ones(len(reduction), dtype=bool)), ("clean", clean_mask)):
        x, y, a = reduction[mask], lift[mask], after_accuracy[mask]
        shorter = x > 0
        output[label] = {
            "n": int(mask.sum()),
            "pearson_reduction_vs_lift": corr(x, y),
            "spearman_reduction_vs_lift": corr(rankdata(x), rankdata(y)),
            "pearson_reduction_vs_after_accuracy": corr(x, a),
            "mean_reduction": float(x.mean()),
            "median_reduction": float(np.median(x)),
            "shorter_fraction": float(shorter.mean()),
            "mean_lift": float(y.mean()),
            "shorter_group_mean_lift": float(y[shorter].mean()),
            "not_shorter_group_mean_lift": float(y[~shorter].mean()),
            "shorter_group_improved_fraction": float((y[shorter] > 0).mean()),
            "not_shorter_group_improved_fraction": float((y[~shorter] > 0).mean()),
        }
    return output


def analyze_horizon(horizon: int) -> dict:
    after_full_sum = np.zeros((EXPECTED_PREFIXES, 8), dtype=np.int64)
    after_cont_sum = np.zeros((EXPECTED_PREFIXES, 8), dtype=np.int64)
    after_correct = np.zeros((EXPECTED_PREFIXES, 8), dtype=np.int16)
    after_count = np.zeros((EXPECTED_PREFIXES, 8), dtype=np.int16)
    after_has_length = np.zeros((EXPECTED_PREFIXES, 8), dtype=bool)
    pooled_full = []
    pooled_cont = []
    finish_counts = Counter()
    replicate_summaries = []

    for replicate in range(8):
        seen = set()
        rep_full = []
        rep_cont = []
        rep_correct = 0
        rep_finish = Counter()
        files = rep_score_files(horizon, replicate)
        for path in files:
            for row in iter_jsonl(path):
                dataset_id = int(row["dataset_index"])
                if dataset_id not in id_to_pos:
                    raise RuntimeError(f"h{horizon} rep{replicate}: unknown dataset_index {dataset_id}")
                pos = id_to_pos[dataset_id]
                bridge_index = int(row["rollout_index"])
                key = (pos, bridge_index)
                if key in seen:
                    raise RuntimeError(f"h{horizon} rep{replicate}: duplicate key {key}")
                seen.add(key)
                full = int(row["full_trace_token_count"])
                cont = int(row.get("student_continuation_token_count", row["completion_token_count"]))
                correct = int(bool(row["correct"]))
                finish = str(row.get("finish_reason", "missing"))
                bridge_tokens = int(row["source_bridge_token_count"])
                if bridge_tokens != horizon:
                    raise RuntimeError(f"h{horizon} rep{replicate}: bridge length {bridge_tokens}")
                after_full_sum[pos, bridge_index] += full
                after_cont_sum[pos, bridge_index] += cont
                after_correct[pos, bridge_index] += correct
                after_count[pos, bridge_index] += 1
                after_has_length[pos, bridge_index] |= finish == "length"
                rep_full.append(full)
                rep_cont.append(cont)
                rep_correct += correct
                rep_finish[finish] += 1
        if len(seen) != EXPECTED_ROLLOUTS:
            raise RuntimeError(f"h{horizon} rep{replicate}: coverage {len(seen)} != {EXPECTED_ROLLOUTS}")
        pooled_full.extend(rep_full)
        pooled_cont.extend(rep_cont)
        finish_counts.update(rep_finish)
        replicate_summaries.append(
            {
                "replicate": replicate,
                "source_files": len(files),
                "rollouts": len(rep_full),
                "mean_full": statistics.fmean(rep_full),
                "median_full": statistics.median(rep_full),
                "mean_continuation": statistics.fmean(rep_cont),
                "accuracy": rep_correct / len(rep_full),
                "finish_reasons": dict(sorted(rep_finish.items())),
            }
        )

    if not np.all(after_count == 8):
        raise RuntimeError(f"h{horizon}: aggregate bridge coverage is not exactly 8")
    after_full_mean = after_full_sum / 8.0
    after_cont_mean = after_cont_sum / 8.0
    after_acc = after_correct / 8.0
    bridge_reduction = direct_full_mean[:, None] - after_full_mean
    bridge_cont_reduction = direct_cont_mean[:, None] - after_cont_mean
    bridge_lift = after_acc - direct_acc[:, None]
    bridge_clean = ~(direct_has_length[:, None] | after_has_length)

    prefix_after_full = after_full_sum.sum(axis=1) / 64.0
    prefix_after_acc = after_correct.sum(axis=1) / 64.0
    prefix_reduction = direct_full_mean - prefix_after_full
    prefix_lift = prefix_after_acc - direct_acc
    prefix_clean = ~(direct_has_length | after_has_length.any(axis=1))

    lift_steps = after_correct - direct_correct[:, None]
    lift_hist = {str(step): int((lift_steps == step).sum()) for step in range(-8, 9)}
    after_correct_hist = {str(k): int((after_correct == k).sum()) for k in range(9)}
    bridge_pass_all_8 = after_correct == 8
    pass_all_8_bridges_per_prefix = bridge_pass_all_8.sum(axis=1)
    any_bridge_pass_all_8 = pass_all_8_bridges_per_prefix > 0
    all_bridges_pass_all_8 = pass_all_8_bridges_per_prefix == 8
    student_pass_all_8 = direct_correct == 8
    lift_bucket_problem_counts = {}
    for step in range(-8, 9):
        mask = lift_steps == step
        positions = np.flatnonzero(mask.any(axis=1))
        problem_ids = {direct_problem_ids[int(pos)] for pos in positions}
        lift_bucket_problem_counts[str(step)] = {
            "bridge_count": int(mask.sum()),
            "unique_prefixes": int(positions.size),
            "unique_problems": len(problem_ids),
        }
    lift_bucket_trace = {}
    for step in range(-8, 9):
        mask = lift_steps == step
        clean = mask & bridge_clean
        x = bridge_reduction[mask]
        y = bridge_reduction[clean]
        lift_bucket_trace[str(step)] = {
            "n": int(mask.sum()),
            "mean_reduction": float(x.mean()),
            "median_reduction": float(np.median(x)),
            "clean_n": int(clean.sum()),
            "clean_mean_reduction": float(y.mean()),
            "clean_median_reduction": float(np.median(y)),
            "clean_shorter_fraction": float((y > 0).mean()),
        }

    zero_mask = direct_correct == 0
    zero_after_correct = after_correct[zero_mask]
    zero_bridge_hist = {str(k): int((zero_after_correct == k).sum()) for k in range(9)}
    zero_any_improved = (zero_after_correct > 0).any(axis=1)
    zero_num_improved_bridges = (zero_after_correct > 0).sum(axis=1)

    perfect_mask = zero_after_correct == 8
    perfect_per_prefix = perfect_mask.sum(axis=1)
    perfect_multiplicity = {
        str(k): int((perfect_per_prefix[perfect_per_prefix > 0] == k).sum()) for k in range(1, 9)
    }
    perfect_count = int(perfect_mask.sum())
    perfect_prefix_count = int((perfect_per_prefix > 0).sum())
    perfect_problem_counter = Counter()
    zero_positions = np.flatnonzero(zero_mask)
    for zero_pos, count in enumerate(perfect_per_prefix):
        if count:
            global_pos = int(zero_positions[zero_pos])
            perfect_problem_counter[direct_problem_ids[global_pos]] += int(count)

    random_seed = int(os.environ.get("RANDOM_BRIDGE_SEED", "20260908"))
    random_simulations = int(os.environ.get("RANDOM_BRIDGE_SIMS", "10000"))
    random_rng = np.random.default_rng(random_seed)
    prefix_positions = np.arange(EXPECTED_PREFIXES)[None, :]
    simulated_accuracies = []
    simulated_pass_all_8 = []
    simulated_pass_at_8 = []
    batch_size = 200
    for start in range(0, random_simulations, batch_size):
        batch = min(batch_size, random_simulations - start)
        choices = random_rng.integers(0, 8, size=(batch, EXPECTED_PREFIXES))
        selected_correct = after_correct[prefix_positions, choices]
        simulated_accuracies.extend(selected_correct.sum(axis=1) / (EXPECTED_PREFIXES * 8.0))
        simulated_pass_all_8.extend((selected_correct == 8).mean(axis=1))
        simulated_pass_at_8.extend((selected_correct > 0).mean(axis=1))
    simulated_accuracies = np.asarray(simulated_accuracies)
    simulated_pass_all_8 = np.asarray(simulated_pass_all_8)
    simulated_pass_at_8 = np.asarray(simulated_pass_at_8)
    selected_correct_counts = after_correct[np.arange(EXPECTED_PREFIXES), np.random.default_rng(random_seed).integers(0, 8, EXPECTED_PREFIXES)]
    exact_selection_sd = float(
        np.sqrt(np.var(after_correct, axis=1, ddof=0).sum()) / (EXPECTED_PREFIXES * 8.0)
    )
    selected_pass_all_8 = selected_correct_counts == 8
    selected_pass_at_8 = selected_correct_counts > 0
    exact_pass_all_8_selection_sd = float(
        np.sqrt(np.var(after_correct == 8, axis=1, ddof=0).sum()) / EXPECTED_PREFIXES
    )
    exact_pass_at_8_selection_sd = float(
        np.sqrt(np.var(after_correct > 0, axis=1, ddof=0).sum()) / EXPECTED_PREFIXES
    )

    return {
        "horizon": horizon,
        "strict_coverage": {
            "replicates": 8,
            "rollouts": len(pooled_full),
            "bridge_groups": int(after_count.size),
            "prefixes": EXPECTED_PREFIXES,
        },
        "replicate_summaries": replicate_summaries,
        "pooled": {
            "accuracy": float(after_correct.sum() / after_count.sum()),
            "accuracy_lift_vs_direct": float(after_correct.sum() / after_count.sum() - direct_acc.mean()),
            "full_trace": describe(pooled_full),
            "continuation": describe(pooled_cont),
            "finish_reasons": dict(sorted(finish_counts.items())),
            "stop_only_mean_full": float(np.mean([v for rep in replicate_summaries for v in []])) if False else None,
        },
        "bridge_level": association_summary(
            bridge_reduction.ravel(), bridge_lift.ravel(), after_acc.ravel(), bridge_clean.ravel()
        ),
        "prefix_level": association_summary(prefix_reduction, prefix_lift, prefix_after_acc, prefix_clean),
        "lift_histogram_steps": lift_hist,
        "lift_bucket_problem_counts": lift_bucket_problem_counts,
        "pass_metrics": {
            "bridge_groups": int(after_correct.size),
            "after_correct_count_histogram": after_correct_hist,
            "pass_all_8_count": int((after_correct == 8).sum()),
            "pass_all_8_fraction": float((after_correct == 8).mean()),
            "pass_at_8_count": int((after_correct > 0).sum()),
            "pass_at_8_fraction": float((after_correct > 0).mean()),
            "improved_vs_direct_count": int((lift_steps > 0).sum()),
            "improved_vs_direct_fraction": float((lift_steps > 0).mean()),
            "unchanged_vs_direct_count": int((lift_steps == 0).sum()),
            "degraded_vs_direct_count": int((lift_steps < 0).sum()),
        },
        "prefix_pass_all_8_relationships": {
            "prefixes": EXPECTED_PREFIXES,
            "pass_all_8_bridge_count_histogram": {
                str(k): int((pass_all_8_bridges_per_prefix == k).sum())
                for k in range(9)
            },
            "any_bridge_pass_all_8_count": int(any_bridge_pass_all_8.sum()),
            "any_bridge_pass_all_8_fraction": float(any_bridge_pass_all_8.mean()),
            "all_bridges_pass_all_8_count": int(all_bridges_pass_all_8.sum()),
            "all_bridges_pass_all_8_fraction": float(all_bridges_pass_all_8.mean()),
            "all_bridges_given_any_bridge_fraction": float(
                all_bridges_pass_all_8.sum() / any_bridge_pass_all_8.sum()
            ),
            "any_vs_all_bridges": binary_association(
                any_bridge_pass_all_8, all_bridges_pass_all_8
            ),
            "student_pass_all_8_count": int(student_pass_all_8.sum()),
            "student_vs_any_bridge": binary_association(
                student_pass_all_8, any_bridge_pass_all_8
            ),
            "student_vs_all_bridges": binary_association(
                student_pass_all_8, all_bridges_pass_all_8
            ),
        },
        "lift_bucket_trace": lift_bucket_trace,
        "zero_baseline": {
            "prefixes": int(zero_mask.sum()),
            "bridges": int(zero_after_correct.size),
            "bridge_after_correct_count_histogram": zero_bridge_hist,
            "bridges_improved": int((zero_after_correct > 0).sum()),
            "bridges_improved_fraction": float((zero_after_correct > 0).mean()),
            "prefixes_any_bridge_improves": int(zero_any_improved.sum()),
            "prefixes_any_bridge_improves_fraction": float(zero_any_improved.mean()),
            "prefix_histogram_number_of_improved_bridges": {
                str(k): int((zero_num_improved_bridges == k).sum()) for k in range(9)
            },
            "pooled_after_accuracy": float(zero_after_correct.sum() / (zero_after_correct.size * 8)),
        },
        "perfect_plus_100pp": {
            "bridges": perfect_count,
            "distinct_prefixes": perfect_prefix_count,
            "mean_per_represented_prefix": perfect_count / perfect_prefix_count if perfect_prefix_count else None,
            "multiplicity_histogram_prefixes": perfect_multiplicity,
            "prefixes_with_multiple": int((perfect_per_prefix >= 2).sum()),
            "bridges_from_multiple_prefixes": int(perfect_per_prefix[perfect_per_prefix >= 2].sum()),
            "prefixes_all_8": int((perfect_per_prefix == 8).sum()),
            "bridges_from_all_8_prefixes": int(perfect_per_prefix[perfect_per_prefix == 8].sum()),
            "unique_problems": len(perfect_problem_counter),
            "mean_per_represented_problem": (
                perfect_count / len(perfect_problem_counter) if perfect_problem_counter else None
            ),
            "multiplicity_histogram_problems": {
                str(k): int(sum(value == k for value in perfect_problem_counter.values()))
                for k in sorted(set(perfect_problem_counter.values()))
            },
            "problem_ids": sorted(perfect_problem_counter),
        },
        "random_one_bridge_per_prefix": {
            "seed": random_seed,
            "simulations": random_simulations,
            "rollouts_per_selection": EXPECTED_PREFIXES * 8,
            "all_bridges_expected_accuracy": float(after_correct.mean() / 8.0),
            "single_seed_correct_count": int(selected_correct_counts.sum()),
            "single_seed_accuracy": float(selected_correct_counts.sum() / (EXPECTED_PREFIXES * 8.0)),
            "single_seed_lift_vs_direct": float(
                selected_correct_counts.sum() / (EXPECTED_PREFIXES * 8.0) - direct_acc.mean()
            ),
            "exact_selection_sd": exact_selection_sd,
            "simulation_mean_accuracy": float(simulated_accuracies.mean()),
            "simulation_std_accuracy": float(simulated_accuracies.std(ddof=1)),
            "simulation_p2_5_accuracy": float(np.quantile(simulated_accuracies, 0.025)),
            "simulation_p97_5_accuracy": float(np.quantile(simulated_accuracies, 0.975)),
            "simulation_min_accuracy": float(simulated_accuracies.min()),
            "simulation_max_accuracy": float(simulated_accuracies.max()),
            "pass_all_8": {
                "all_bridges_expected_fraction": float((after_correct == 8).mean()),
                "single_seed_count": int(selected_pass_all_8.sum()),
                "single_seed_fraction": float(selected_pass_all_8.mean()),
                "single_seed_delta_vs_direct": float(
                    selected_pass_all_8.mean() - (direct_correct == 8).mean()
                ),
                "exact_selection_sd": exact_pass_all_8_selection_sd,
                "simulation_mean_fraction": float(simulated_pass_all_8.mean()),
                "simulation_std_fraction": float(simulated_pass_all_8.std(ddof=1)),
                "simulation_p2_5_fraction": float(np.quantile(simulated_pass_all_8, 0.025)),
                "simulation_p97_5_fraction": float(np.quantile(simulated_pass_all_8, 0.975)),
                "simulation_min_fraction": float(simulated_pass_all_8.min()),
                "simulation_max_fraction": float(simulated_pass_all_8.max()),
            },
            "pass_at_8": {
                "all_bridges_expected_fraction": float((after_correct > 0).mean()),
                "single_seed_count": int(selected_pass_at_8.sum()),
                "single_seed_fraction": float(selected_pass_at_8.mean()),
                "single_seed_delta_vs_direct": float(
                    selected_pass_at_8.mean() - (direct_correct > 0).mean()
                ),
                "exact_selection_sd": exact_pass_at_8_selection_sd,
                "simulation_mean_fraction": float(simulated_pass_at_8.mean()),
                "simulation_std_fraction": float(simulated_pass_at_8.std(ddof=1)),
                "simulation_p2_5_fraction": float(np.quantile(simulated_pass_at_8, 0.025)),
                "simulation_p97_5_fraction": float(np.quantile(simulated_pass_at_8, 0.975)),
                "simulation_min_fraction": float(simulated_pass_at_8.min()),
                "simulation_max_fraction": float(simulated_pass_at_8.max()),
            },
        },
    }


requested_horizons = tuple(
    int(value) for value in os.environ.get("ANALYSIS_HORIZONS", "32,64,128,256").split(",")
)

report = {
    "definitions": {
        "whole_trace": "response prefix + bridge + generated continuation",
        "reduction": "direct Student mean whole-trace length minus after-bridge mean; positive means shorter",
        "clean": "comparison contains no length-finished rollout on either side",
        "bridge_level": "one (prefix, source bridge) group averaged over eight replicate continuations",
        "prefix_level": "all 8 bridges x 8 replicates averaged for one prefix",
    },
    "direct": {
        "rollouts": len(direct_rows),
        "prefixes": EXPECTED_PREFIXES,
        "accuracy": float(direct_acc.mean()),
        "full_trace": describe(direct_full_values),
        "continuation": describe(direct_cont_values),
        "finish_reasons": dict(sorted(direct_finish.items())),
        "zero_of_8_prefixes": int((direct_correct == 0).sum()),
        "pass_all_8_count": int((direct_correct == 8).sum()),
        "pass_all_8_fraction": float((direct_correct == 8).mean()),
        "pass_at_8_count": int((direct_correct > 0).sum()),
        "pass_at_8_fraction": float((direct_correct > 0).mean()),
        "correct_count_histogram": {
            str(k): int((direct_correct == k).sum()) for k in range(9)
        },
        "unique_problems": len(set(direct_problem_ids)),
    },
    "horizons": {str(horizon): analyze_horizon(horizon) for horizon in requested_horizons},
}
if os.environ.get("PREFIX_PASS_ALL_REL_ONLY"):
    output = {
        "direct": {
            "prefixes": report["direct"]["prefixes"],
            "pass_all_8_count": report["direct"]["pass_all_8_count"],
            "pass_all_8_fraction": report["direct"]["pass_all_8_fraction"],
        },
        "horizons": {
            horizon: values["prefix_pass_all_8_relationships"]
            for horizon, values in report["horizons"].items()
        },
    }
    print(json.dumps(output, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
elif os.environ.get("RANDOM_BRIDGE_ONLY"):
    output = {
        "direct": {
            "prefixes": report["direct"]["prefixes"],
            "rollouts": report["direct"]["rollouts"],
            "accuracy": report["direct"]["accuracy"],
        },
        "horizons": {
            horizon: values["random_one_bridge_per_prefix"]
            for horizon, values in report["horizons"].items()
        },
    }
    print(json.dumps(output, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
elif os.environ.get("LIFT_DEDUP_ONLY"):
    output = {
        "direct": {
            "prefixes": report["direct"]["prefixes"],
            "unique_problems": report["direct"]["unique_problems"],
        },
        "horizons": {
            horizon: values["lift_bucket_problem_counts"]
            for horizon, values in report["horizons"].items()
        },
    }
    print(json.dumps(output, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
elif os.environ.get("PERFECT_ONLY"):
    problem_sets = {
        horizon: set(values["perfect_plus_100pp"]["problem_ids"])
        for horizon, values in report["horizons"].items()
    }
    membership = Counter()
    for problem_id in set().union(*problem_sets.values()):
        key = "+".join(
            horizon for horizon in sorted(problem_sets, key=int)
            if problem_id in problem_sets[horizon]
        )
        membership[key] += 1
    output = {
        "direct": {
            "prefixes": report["direct"]["prefixes"],
            "unique_problems": report["direct"]["unique_problems"],
        },
        "horizons": {
            horizon: values["perfect_plus_100pp"]
            for horizon, values in report["horizons"].items()
        },
        "exact_problem_horizon_membership": dict(sorted(membership.items())),
    }
    print(json.dumps(output, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
elif os.environ.get("PASS_ONLY"):
    output = {
        "direct": {
            key: value for key, value in report["direct"].items()
            if key in {
                "prefixes", "accuracy", "pass_all_8_count", "pass_all_8_fraction",
                "pass_at_8_count", "pass_at_8_fraction", "correct_count_histogram",
            }
        },
        "horizons": {
            horizon: values["pass_metrics"]
            for horizon, values in report["horizons"].items()
        },
    }
    print(json.dumps(output, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
elif os.environ.get("COMPACT"):
    compact = {
        "direct": report["direct"],
        "horizons": {
            horizon: {
                "replicate_summaries": values["replicate_summaries"],
                "pooled": values["pooled"],
                "bridge_level": values["bridge_level"],
                "prefix_level": values["prefix_level"],
                "lift_histogram_steps": values["lift_histogram_steps"],
                "lift_bucket_problem_counts": values["lift_bucket_problem_counts"],
                "pass_metrics": values["pass_metrics"],
                "random_one_bridge_per_prefix": values["random_one_bridge_per_prefix"],
                "lift_bucket_trace": values["lift_bucket_trace"],
                "zero_baseline": values["zero_baseline"],
                "perfect_plus_100pp": values["perfect_plus_100pp"],
            }
            for horizon, values in report["horizons"].items()
        },
    }
    print(json.dumps(compact, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
else:
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
