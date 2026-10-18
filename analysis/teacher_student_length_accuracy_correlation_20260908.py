#!/usr/bin/env python3
"""Paired prefix-level Teacher-vs-Student length/accuracy association."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path("/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train")
SUFFIX = Path("analysis_correctness_length_single_job/scores.jsonl")
STUDENT = ROOT / "student_from_anchor_prefix_qwen3_1_7b_n8_seed20260827" / SUFFIX
TEACHER = ROOT / "teacher_full_qwen3_14b_n8_seed20260826" / SUFFIX


def load(path: Path) -> dict[int, dict]:
    grouped = defaultdict(lambda: {"n": 0, "full": 0, "cont": 0, "correct": 0, "has_length": False})
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            target = grouped[int(row["dataset_index"])]
            continuation = row.get("student_completion_token_count")
            if continuation is None:
                continuation = row.get("teacher_completion_token_count")
            if continuation is None:
                continuation = row["completion_token_count"]
            target["n"] += 1
            target["full"] += int(row["full_trace_token_count"])
            target["cont"] += int(continuation)
            target["correct"] += int(bool(row["correct"]))
            target["has_length"] |= row.get("finish_reason") == "length"
    if any(value["n"] != 8 for value in grouped.values()):
        raise RuntimeError(f"Expected exactly 8 rollouts per prefix: {path}")
    return grouped


def corr_summary(x: np.ndarray, y: np.ndarray) -> dict:
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    slope = np.cov(x, y, ddof=0)[0, 1] / np.var(x)
    return {
        "n": int(len(x)),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "accuracy_pp_per_100_extra_tokens": float(slope * 10000.0),
    }


def partial_corr(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> dict:
    design = np.column_stack([np.ones(len(x)), controls])
    x_residual = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y_residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    result = stats.pearsonr(x_residual, y_residual)
    return {"r": float(result.statistic), "p": float(result.pvalue)}


def describe(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "min": float(values.min()),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def difference_histogram(values: np.ndarray) -> list[dict]:
    specifications = [
        ("< -1000", values < -1000),
        ("[-1000, -500)", (values >= -1000) & (values < -500)),
        ("[-500, -250)", (values >= -500) & (values < -250)),
        ("[-250, -100)", (values >= -250) & (values < -100)),
        ("[-100, 0)", (values >= -100) & (values < 0)),
        ("0", values == 0),
        ("(0, 100)", (values > 0) & (values < 100)),
        ("[100, 250)", (values >= 100) & (values < 250)),
        ("[250, 500)", (values >= 250) & (values < 500)),
        ("[500, 1000)", (values >= 500) & (values < 1000)),
        (">= 1000", values >= 1000),
    ]
    return [
        {"bin": label, "prefixes": int(selected.sum()), "fraction": float(selected.mean())}
        for label, selected in specifications
    ]


def difference_accuracy_histogram(values: np.ndarray, accuracy_delta: np.ndarray) -> list[dict]:
    specifications = [
        ("< -1000", values < -1000),
        ("[-1000, -500)", (values >= -1000) & (values < -500)),
        ("[-500, -250)", (values >= -500) & (values < -250)),
        ("[-250, -100)", (values >= -250) & (values < -100)),
        ("[-100, 0)", (values >= -100) & (values < 0)),
        ("0", values == 0),
        ("(0, 100)", (values > 0) & (values < 100)),
        ("[100, 250)", (values >= 100) & (values < 250)),
        ("[250, 500)", (values >= 250) & (values < 500)),
        ("[500, 1000)", (values >= 500) & (values < 1000)),
        (">= 1000", values >= 1000),
    ]
    output = []
    for label, selected in specifications:
        n = int(selected.sum())
        output.append(
            {
                "bin": label,
                "prefixes": n,
                "accuracy_decreased": int((selected & (accuracy_delta < 0)).sum()),
                "accuracy_equal": int((selected & (accuracy_delta == 0)).sum()),
                "accuracy_increased": int((selected & (accuracy_delta > 0)).sum()),
                "accuracy_increased_fraction": float((accuracy_delta[selected] > 0).mean()) if n else None,
                "mean_accuracy_lift_pp": float(accuracy_delta[selected].mean() * 100) if n else None,
            }
        )
    return output


student = load(STUDENT)
teacher = load(TEACHER)
ids = sorted(set(student) & set(teacher))
if len(ids) != 13244 or set(student) != set(teacher):
    raise RuntimeError(f"Unexpected paired coverage: {len(ids)}")

student_full = np.asarray([student[i]["full"] / 8 for i in ids], dtype=float)
student_cont = np.asarray([student[i]["cont"] / 8 for i in ids], dtype=float)
student_acc = np.asarray([student[i]["correct"] / 8 for i in ids], dtype=float)
teacher_full = np.asarray([teacher[i]["full"] / 8 for i in ids], dtype=float)
teacher_cont = np.asarray([teacher[i]["cont"] / 8 for i in ids], dtype=float)
teacher_acc = np.asarray([teacher[i]["correct"] / 8 for i in ids], dtype=float)
clean = np.asarray([not (student[i]["has_length"] or teacher[i]["has_length"]) for i in ids])

full_delta = teacher_full - student_full
cont_delta = teacher_cont - student_cont
accuracy_lift = teacher_acc - student_acc


def analyze(mask: np.ndarray) -> dict:
    x = full_delta[mask]
    xc = cont_delta[mask]
    y = accuracy_lift[mask]
    student_correct_count = np.rint(student_acc[mask] * 8).astype(int)
    teacher_correct_count = np.rint(teacher_acc[mask] * 8).astype(int)
    longer = x > 0
    improved = y > 0
    controls = np.column_stack([student_full[mask], student_acc[mask]])
    relative_cont_delta = xc / student_cont[mask]
    paired_t = stats.ttest_1samp(xc, 0.0)
    wilcoxon = stats.wilcoxon(xc)
    at_least_100 = xc >= 100
    unchanged_at_least_100 = at_least_100 & (y == 0)
    improved_at_least_100 = at_least_100 & (y > 0)
    unchanged_accuracy_histogram = {
        str(correct_count): int((unchanged_at_least_100 & (student_correct_count == correct_count)).sum())
        for correct_count in range(9)
    }
    improved_lift_histogram = {
        str(step): int((improved_at_least_100 & ((teacher_correct_count - student_correct_count) == step)).sum())
        for step in range(1, 9)
    }
    improved_transition_histogram = {
        f"{student_count}->{teacher_count}": int(
            (
                improved_at_least_100
                & (student_correct_count == student_count)
                & (teacher_correct_count == teacher_count)
            ).sum()
        )
        for student_count in range(8)
        for teacher_count in range(student_count + 1, 9)
        if int(
            (
                improved_at_least_100
                & (student_correct_count == student_count)
                & (teacher_correct_count == teacher_count)
            ).sum()
        )
    }
    improved_length_by_lift = {}
    for step in range(1, 9):
        selected = improved_at_least_100 & ((teacher_correct_count - student_correct_count) == step)
        values = xc[selected]
        improved_length_by_lift[str(step)] = {
            "accuracy_lift_pp": step * 12.5,
            "prefixes": int(selected.sum()),
            "mean_teacher_minus_student_continuation_tokens": float(values.mean()),
            "p25_teacher_minus_student_continuation_tokens": float(np.quantile(values, 0.25)),
            "median_teacher_minus_student_continuation_tokens": float(np.median(values)),
            "p75_teacher_minus_student_continuation_tokens": float(np.quantile(values, 0.75)),
        }
    improved_step = teacher_correct_count[improved_at_least_100] - student_correct_count[improved_at_least_100]
    length_per_lift_step = float(
        np.cov(improved_step, xc[improved_at_least_100], ddof=0)[0, 1]
        / np.var(improved_step)
    )

    length_direction = np.sign(x).astype(int)
    accuracy_direction = np.sign(y).astype(int)
    direction_labels = {-1: "decrease", 0: "equal", 1: "increase"}
    contingency = np.asarray(
        [
            [int(((length_direction == length_sign) & (accuracy_direction == accuracy_sign)).sum())
             for accuracy_sign in (-1, 0, 1)]
            for length_sign in (-1, 0, 1)
        ],
        dtype=int,
    )
    nonempty_rows = contingency.sum(axis=1) > 0
    nonempty_cols = contingency.sum(axis=0) > 0
    chi2, chi2_p, _, _ = stats.chi2_contingency(
        contingency[np.ix_(nonempty_rows, nonempty_cols)]
    )
    cramers_v = np.sqrt(
        chi2
        / (
            contingency.sum()
            * min(int(nonempty_rows.sum()) - 1, int(nonempty_cols.sum()) - 1)
        )
    )

    changed = (length_direction != 0) & (accuracy_direction != 0)
    shorter_changed = (length_direction == -1) & changed
    longer_changed = (length_direction == 1) & changed
    shorter_improved = int((shorter_changed & (accuracy_direction == 1)).sum())
    shorter_degraded = int((shorter_changed & (accuracy_direction == -1)).sum())
    longer_improved = int((longer_changed & (accuracy_direction == 1)).sum())
    longer_degraded = int((longer_changed & (accuracy_direction == -1)).sum())
    binary_table = np.asarray(
        [[shorter_improved, shorter_degraded], [longer_improved, longer_degraded]],
        dtype=int,
    )
    binary_odds_ratio, binary_p = stats.fisher_exact(binary_table)

    by_length_direction = {}
    for length_sign in (-1, 0, 1):
        selected = length_direction == length_sign
        if not selected.any():
            continue
        by_length_direction[direction_labels[length_sign]] = {
            "prefixes": int(selected.sum()),
            "mean_accuracy_lift_pp": float(y[selected].mean() * 100),
            "accuracy_decreased_fraction": float((accuracy_direction[selected] == -1).mean()),
            "accuracy_equal_fraction": float((accuracy_direction[selected] == 0).mean()),
            "accuracy_increased_fraction": float((accuracy_direction[selected] == 1).mean()),
        }

    by_accuracy_direction = {}
    for accuracy_sign in (-1, 0, 1):
        selected = accuracy_direction == accuracy_sign
        if not selected.any():
            continue
        by_accuracy_direction[direction_labels[accuracy_sign]] = {
            "prefixes": int(selected.sum()),
            "mean_teacher_minus_student_full_tokens": float(x[selected].mean()),
            "length_decreased_fraction": float((length_direction[selected] == -1).mean()),
            "length_equal_fraction": float((length_direction[selected] == 0).mean()),
            "length_increased_fraction": float((length_direction[selected] == 1).mean()),
        }

    buckets = {}
    lift_steps = np.rint(y * 8).astype(int)
    for step in range(-8, 9):
        selected = lift_steps == step
        if not selected.any():
            continue
        buckets[str(step)] = {
            "accuracy_lift_pp": step * 12.5,
            "prefixes": int(selected.sum()),
            "mean_teacher_minus_student_full_tokens": float(x[selected].mean()),
            "median_teacher_minus_student_full_tokens": float(np.median(x[selected])),
            "teacher_shorter_count": int((x[selected] < 0).sum()),
            "teacher_equal_count": int((x[selected] == 0).sum()),
            "teacher_longer_count": int((x[selected] > 0).sum()),
            "teacher_shorter_fraction": float((x[selected] < 0).mean()),
            "teacher_longer_fraction": float((x[selected] > 0).mean()),
            "trace_delta_distribution": describe(x[selected]),
            "trace_delta_histogram": difference_histogram(x[selected]),
        }

    return {
        "prefixes": int(mask.sum()),
        "student_continuation_per_prefix_mean_distribution": describe(student_cont[mask]),
        "teacher_continuation_per_prefix_mean_distribution": describe(teacher_cont[mask]),
        "teacher_minus_student_continuation_distribution": describe(xc),
        "teacher_minus_student_relative_continuation_distribution": describe(relative_cont_delta),
        "teacher_minus_student_continuation_histogram": difference_histogram(xc),
        "continuation_bins_by_accuracy_change": difference_accuracy_histogram(xc, y),
        "continuation_at_least_100_tokens_longer": {
            "prefixes": int((xc >= 100).sum()),
            "accuracy_decreased": int(((xc >= 100) & (y < 0)).sum()),
            "accuracy_equal": int(((xc >= 100) & (y == 0)).sum()),
            "accuracy_increased": int(((xc >= 100) & (y > 0)).sum()),
            "accuracy_increased_fraction": float((y[xc >= 100] > 0).mean()),
            "mean_accuracy_lift_pp": float(y[xc >= 100].mean() * 100),
            "unchanged_shared_correct_count_histogram": unchanged_accuracy_histogram,
            "improved_lift_steps_histogram": improved_lift_histogram,
            "improved_student_to_teacher_correct_count_histogram": improved_transition_histogram,
            "improved_length_by_lift_step": improved_length_by_lift,
            "improved_length_delta_vs_accuracy_lift": corr_summary(
                xc[improved_at_least_100], y[improved_at_least_100]
            ),
            "extra_continuation_tokens_per_additional_correct_out_of_8": length_per_lift_step,
        },
        "continuation_difference_test": {
            "paired_t_statistic": float(paired_t.statistic),
            "paired_t_p": float(paired_t.pvalue),
            "wilcoxon_statistic": float(wilcoxon.statistic),
            "wilcoxon_p": float(wilcoxon.pvalue),
        },
        "mean_teacher_minus_student_full_tokens": float(x.mean()),
        "median_teacher_minus_student_full_tokens": float(np.median(x)),
        "mean_teacher_minus_student_continuation_tokens": float(xc.mean()),
        "mean_accuracy_lift_pp": float(y.mean() * 100),
        "full_length_delta_vs_accuracy_lift": corr_summary(x, y),
        "continuation_delta_vs_accuracy_lift": corr_summary(xc, y),
        "partial_pearson_controlling_student_length_and_accuracy": partial_corr(x, y, controls),
        "direction_contingency": {
            "rows_length_direction": ["decrease", "equal", "increase"],
            "columns_accuracy_direction": ["decrease", "equal", "increase"],
            "counts": contingency.tolist(),
            "chi_square": float(chi2),
            "p_value": float(chi2_p),
            "cramers_v": float(cramers_v),
            "by_length_direction": by_length_direction,
            "by_accuracy_direction": by_accuracy_direction,
        },
        "changed_only_shorter_vs_longer_and_improved_vs_degraded": {
            "counts_rows_shorter_longer_columns_improved_degraded": binary_table.tolist(),
            "shorter_accuracy_improved_fraction": float(shorter_improved / (shorter_improved + shorter_degraded)),
            "longer_accuracy_improved_fraction": float(longer_improved / (longer_improved + longer_degraded)),
            "risk_difference": float(
                shorter_improved / (shorter_improved + shorter_degraded)
                - longer_improved / (longer_improved + longer_degraded)
            ),
            "odds_ratio_improvement_shorter_vs_longer": float(binary_odds_ratio),
            "fisher_exact_p": float(binary_p),
        },
        "teacher_longer_group": {
            "prefixes": int(longer.sum()),
            "fraction": float(longer.mean()),
            "mean_accuracy_lift_pp": float(y[longer].mean() * 100),
            "accuracy_improves_fraction": float(improved[longer].mean()),
        },
        "teacher_not_longer_group": {
            "prefixes": int((~longer).sum()),
            "fraction": float((~longer).mean()),
            "mean_accuracy_lift_pp": float(y[~longer].mean() * 100),
            "accuracy_improves_fraction": float(improved[~longer].mean()),
        },
        "accuracy_improved_group": {
            "prefixes": int(improved.sum()),
            "mean_teacher_minus_student_full_tokens": float(x[improved].mean()),
            "teacher_longer_fraction": float(longer[improved].mean()),
            "student_full_trace_distribution": describe(student_full[mask][improved]),
            "teacher_full_trace_distribution": describe(teacher_full[mask][improved]),
            "teacher_minus_student_full_trace_distribution": describe(x[improved]),
            "teacher_minus_student_relative_full_trace_distribution": describe(
                x[improved] / student_full[mask][improved]
            ),
            "full_trace_difference_histogram": difference_histogram(x[improved]),
            "full_trace_delta_vs_accuracy_lift": corr_summary(x[improved], y[improved]),
        },
        "accuracy_not_improved_group": {
            "prefixes": int((~improved).sum()),
            "mean_teacher_minus_student_full_tokens": float(x[~improved].mean()),
            "teacher_longer_fraction": float(longer[~improved].mean()),
        },
        "accuracy_lift_buckets": buckets,
    }


report = {
    "definitions": {
        "unit": "one anchor prefix; both models have 8 rollouts",
        "length_delta": "Teacher mean trace tokens minus Student mean trace tokens; positive means Teacher is longer",
        "accuracy_lift": "Teacher accuracy minus Student accuracy for the same prefix",
        "clean": "exclude a prefix if either model has any finish_reason=length rollout",
    },
    "all": analyze(np.ones(len(ids), dtype=bool)),
    "clean": analyze(clean),
}

print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
