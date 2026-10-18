#!/usr/bin/env python3
"""Analyze where teacher-rejected sampled tokens occur and their correctness association."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import math
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np


CASE_PATTERN = re.compile(r'window\.__dmGapReceive\("(\d+)","([A-Za-z0-9+/=]+)"\);\s*')
INDEX_PATTERN = re.compile(r"const INDEX=(.*);\nconst PAGE=", re.DOTALL)


def decode_gap(path: Path) -> tuple[int, np.ndarray]:
    match = CASE_PATTERN.fullmatch(path.read_text(encoding="utf-8"))
    if match is None:
        raise RuntimeError(f"Invalid case payload: {path}")
    raw = gzip.decompress(base64.b64decode(match.group(2)))
    count, scale = struct.unpack_from("<If", raw, 0)
    gaps = np.empty(count, dtype=np.float64)
    offset = 8
    for index in range(count):
        quantized, text_length = struct.unpack_from("<hH", raw, offset)
        offset += 4
        gaps[index] = quantized / 32767.0 * scale
        offset += text_length
    if offset != len(raw):
        raise RuntimeError(f"Trailing or missing payload bytes: {path}")
    return int(match.group(1)), gaps


def percentile_ci(values: np.ndarray, rng: np.random.Generator, reps: int = 20000) -> list[float]:
    means = np.empty(reps, dtype=np.float64)
    for index in range(reps):
        means[index] = rng.choice(values, len(values), replace=True).mean()
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def two_group_bootstrap(
    wrong: np.ndarray, correct: np.ndarray, rng: np.random.Generator, reps: int = 20000
) -> list[float]:
    differences = np.empty(reps, dtype=np.float64)
    for index in range(reps):
        differences[index] = (
            rng.choice(wrong, len(wrong), replace=True).mean()
            - rng.choice(correct, len(correct), replace=True).mean()
        )
    return [float(value) for value in np.quantile(differences, [0.025, 0.975])]


def sign_flip_pvalue(values: np.ndarray, rng: np.random.Generator, reps: int = 100000) -> float:
    observed = abs(float(values.mean()))
    exceed = 0
    batch = 1000
    for start in range(0, reps, batch):
        size = min(batch, reps - start)
        signs = rng.integers(0, 2, size=(size, len(values)), dtype=np.int8) * 2 - 1
        simulated = np.abs((signs * values).mean(axis=1))
        exceed += int((simulated >= observed).sum())
    return (exceed + 1) / (reps + 1)


def paired_comparison(
    left: np.ndarray, right: np.ndarray, rng: np.random.Generator
) -> dict[str, Any]:
    difference = left - right
    return {
        "left_rollout_mean_fraction": float(left.mean()),
        "right_rollout_mean_fraction": float(right.mean()),
        "left_minus_right": float(difference.mean()),
        "bootstrap_95ci": percentile_ci(difference, rng),
        "paired_sign_flip_p": sign_flip_pvalue(difference, rng),
    }


def label_permutation_pvalue(
    values: np.ndarray, wrong_mask: np.ndarray, rng: np.random.Generator, reps: int = 50000
) -> float:
    wrong_count = int(wrong_mask.sum())
    observed = abs(float(values[wrong_mask].mean() - values[~wrong_mask].mean()))
    exceed = 0
    for _ in range(reps):
        order = rng.permutation(len(values))
        difference = values[order[:wrong_count]].mean() - values[order[wrong_count:]].mean()
        exceed += abs(float(difference)) >= observed
    return (exceed + 1) / (reps + 1)


def auc_for_wrong(values: np.ndarray, wrong_mask: np.ndarray) -> float:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    n_positive = int(wrong_mask.sum())
    n_negative = len(values) - n_positive
    return float((ranks[wrong_mask].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))


def logistic_adjusted(
    fraction: np.ndarray, wrong: np.ndarray, lengths: np.ndarray, difficulty: np.ndarray
) -> dict[str, float]:
    log_length = np.log(lengths)
    log_length = (log_length - log_length.mean()) / log_length.std()
    difficulty_z = (difficulty - difficulty.mean()) / difficulty.std()
    x = np.column_stack([np.ones(len(fraction)), fraction * 100.0, log_length, difficulty_z])
    y = wrong.astype(np.float64)
    beta = np.zeros(x.shape[1], dtype=np.float64)
    for _ in range(100):
        linear = np.clip(x @ beta, -30, 30)
        probability = 1.0 / (1.0 + np.exp(-linear))
        weight = np.maximum(probability * (1.0 - probability), 1e-9)
        hessian = x.T @ (weight[:, None] * x)
        step = np.linalg.solve(hessian, x.T @ (y - probability))
        largest_step = float(np.max(np.abs(step)))
        if largest_step > 5.0:
            step *= 5.0 / largest_step
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    covariance = np.linalg.inv(hessian)
    standard_error = math.sqrt(float(covariance[1, 1]))
    z_value = float(beta[1] / standard_error)
    p_value = math.erfc(abs(z_value) / math.sqrt(2.0))
    return {
        "odds_ratio_wrong_per_1pp": float(math.exp(beta[1])),
        "coefficient": float(beta[1]),
        "standard_error": standard_error,
        "wald_z": z_value,
        "wald_p": p_value,
        "adjusted_for": "log(response_tokens), difficulty",
    }


def holm_adjust(pvalues: list[float]) -> list[float]:
    count = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(count, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * pvalues[int(index)])
        adjusted[int(index)] = min(1.0, running)
    return adjusted.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    html = (args.page_root / "index.html").read_text(encoding="utf-8")
    match = INDEX_PATTERN.search(html)
    if match is None:
        raise RuntimeError("Could not locate INDEX payload")
    index_payload: dict[str, Any] = json.loads(match.group(1))
    cases = sorted(index_payload["cases"], key=lambda item: int(item["index"]))
    if len(cases) != 1000:
        raise RuntimeError(f"Expected 1000 cases, found {len(cases)}")

    all_gaps: list[np.ndarray] = []
    for case in cases:
        case_index = int(case["index"])
        decoded_index, gaps = decode_gap(args.page_root / "data" / f"case_{case_index:04d}.js")
        if decoded_index != case_index or len(gaps) != int(case["tokens"]):
            raise RuntimeError(f"Case alignment failure at {case_index}")
        all_gaps.append(gaps)

    global_p95 = float(index_payload["summary"]["abs_gap_quantiles"]["0.95"])
    thresholds = [
        ("page_strong_student", 0.5 * global_p95),
        ("global_p95_abs", global_p95),
        ("strict_2nat", 2.0),
    ]
    wrong = np.asarray([not bool(case["correct"]) for case in cases], dtype=bool)
    lengths = np.asarray([len(gaps) for gaps in all_gaps], dtype=np.float64)
    difficulty = np.asarray([float(case["difficulty"]) for case in cases], dtype=np.float64)
    rng = np.random.default_rng(20260911)

    results: list[dict[str, Any]] = []
    for name, threshold in thresholds:
        fractions = np.empty((len(cases), 10), dtype=np.float64)
        counts = np.zeros(10, dtype=np.int64)
        totals = np.zeros(10, dtype=np.int64)
        for row_index, gaps in enumerate(all_gaps):
            bins = np.minimum(9, np.floor(np.arange(len(gaps)) * 10 / len(gaps)).astype(np.int8))
            rejected = gaps > threshold
            for decile in range(10):
                selected = bins == decile
                fractions[row_index, decile] = rejected[selected].mean()
                counts[decile] += int(rejected[selected].sum())
                totals[decile] += int(selected.sum())

        rollout_fraction = np.asarray([(gaps > threshold).mean() for gaps in all_gaps])
        front = fractions[:, :2].mean(axis=1)
        back = fractions[:, -2:].mean(axis=1)
        first_half = fractions[:, :5].mean(axis=1)
        second_half = fractions[:, 5:].mean(axis=1)
        middle_early = fractions[:, 2:4].mean(axis=1)
        wrong_values = rollout_fraction[wrong]
        correct_values = rollout_fraction[~wrong]
        deciles = []
        for decile in range(10):
            values = fractions[:, decile]
            deciles.append({
                "decile": decile + 1,
                "relative_position": f"{decile * 10}-{(decile + 1) * 10}%",
                "rollout_mean_fraction": float(values.mean()),
                "rollout_mean_fraction_correct": float(values[~wrong].mean()),
                "rollout_mean_fraction_wrong": float(values[wrong].mean()),
                "token_weighted_fraction": float(counts[decile] / totals[decile]),
                "tokens": int(totals[decile]),
            })
        results.append({
            "name": name,
            "definition": f"teacher-rejected token iff g_t > {threshold:.12g}",
            "threshold": threshold,
            "overall_token_fraction": float(sum(int((gaps > threshold).sum()) for gaps in all_gaps) / lengths.sum()),
            "deciles": deciles,
            "front_back": {
                "front_0_20_vs_back_80_100": paired_comparison(front, back, rng),
                "first_0_50_vs_second_50_100": paired_comparison(first_half, second_half, rng),
                "middle_early_20_40_vs_back_80_100": paired_comparison(middle_early, back, rng),
            },
            "accuracy": {
                "correct_rollouts": int((~wrong).sum()),
                "wrong_rollouts": int(wrong.sum()),
                "correct_mean_fraction": float(correct_values.mean()),
                "wrong_mean_fraction": float(wrong_values.mean()),
                "wrong_minus_correct": float(wrong_values.mean() - correct_values.mean()),
                "bootstrap_95ci": two_group_bootstrap(wrong_values, correct_values, rng),
                "label_permutation_p": label_permutation_pvalue(rollout_fraction, wrong, rng),
                "auc_predict_wrong": auc_for_wrong(rollout_fraction, wrong),
                "adjusted_logistic": logistic_adjusted(rollout_fraction, wrong, lengths, difficulty),
            },
        })

    accuracy_p = [item["accuracy"]["label_permutation_p"] for item in results]
    for comparison_name in (
        "front_0_20_vs_back_80_100",
        "first_0_50_vs_second_50_100",
        "middle_early_20_40_vs_back_80_100",
    ):
        pvalues = [item["front_back"][comparison_name]["paired_sign_flip_p"] for item in results]
        for item, adjusted in zip(results, holm_adjust(pvalues), strict=True):
            item["front_back"][comparison_name]["holm_p_across_thresholds"] = adjusted
    for item, adjusted in zip(results, holm_adjust(accuracy_p), strict=True):
        item["accuracy"]["holm_p_across_thresholds"] = adjusted

    output = {
        "status": "DEEPMATH_NONTHINKING_UNRECOGNIZED_TOKEN_ANALYSIS_OK",
        "source": str(args.page_root),
        "metric": "g_t = log pi_S(y_t|s_t) - log pi_T(y_t|s_t)",
        "primary_unit": "rollout",
        "rollouts": len(cases),
        "tokens": int(lengths.sum()),
        "correct": int((~wrong).sum()),
        "wrong": int(wrong.sum()),
        "quantization_note": "Uses int16 visualization payload; threshold error is at most each trace scale/32767.",
        "threshold_results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
