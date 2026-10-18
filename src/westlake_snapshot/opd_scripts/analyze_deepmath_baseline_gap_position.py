#!/usr/bin/env python3
"""Analyze sampled-token log-probability gaps by relative reasoning position."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def metrics(parts: list[np.ndarray]) -> dict[str, Any]:
    values = np.concatenate(parts).astype(np.float32, copy=False)
    absolute = np.abs(values)
    quantile_levels = np.asarray(
        [0.01, 0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99],
        dtype=np.float64,
    )
    quantile_names = [f"p{int(level * 100):02d}" for level in quantile_levels]
    signed_quantiles = np.quantile(values, quantile_levels)
    absolute_quantiles = np.quantile(absolute, quantile_levels)
    return {
        "tokens": int(len(values)),
        "mean_gap": float(values.mean()),
        "mean_abs_gap": float(absolute.mean()),
        "median_gap": float(np.median(values)),
        "p90_abs_gap": float(np.quantile(absolute, 0.90)),
        "p95_abs_gap": float(np.quantile(absolute, 0.95)),
        "gap_quantiles": {
            name: float(value) for name, value in zip(quantile_names, signed_quantiles, strict=True)
        },
        "abs_gap_quantiles": {
            name: float(value) for name, value in zip(quantile_names, absolute_quantiles, strict=True)
        },
        "positive_fraction": float((values > 0).mean()),
        "negative_fraction": float((values < 0).mean()),
        "zero_fraction": float((values == 0).mean()),
    }


def bootstrap_difference(correct: np.ndarray, wrong: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    observed = float(correct.mean() - wrong.mean())
    samples = np.empty(10000, dtype=np.float64)
    for index in range(len(samples)):
        samples[index] = (
            rng.choice(correct, len(correct), replace=True).mean()
            - rng.choice(wrong, len(wrong), replace=True).mean()
        )
    return {
        "correct_minus_wrong": observed,
        "bootstrap_95ci": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
        "unit": "rollout",
        "replicates": len(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.run_root / "merged" / "rollouts.jsonl")
    rows.sort(key=lambda row: int(row["dataset_index"]))
    if len(rows) != 1000:
        raise RuntimeError(f"Expected 1000 rollouts, found {len(rows)}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    if not isinstance(think_end_id, int) or think_end_id < 0:
        raise RuntimeError("Tokenizer does not expose a </think> token ID")

    buckets: dict[str, dict[str, list[np.ndarray]]] = {
        name: defaultdict(list) for name in ("whole_deciles", "reasoning_deciles", "phases")
    }
    trace_metrics: list[dict[str, Any]] = []
    delimiter_count = 0
    total_scored = 0

    for row in rows:
        index = int(row["dataset_index"])
        shard = index % 4
        score_dir = args.score_root / f"shard_{shard}"
        with np.load(score_dir / "student_logprobs" / f"example_{index:04d}.npz") as value:
            student = value["sample_logprob"].astype(np.float32)
        with np.load(score_dir / "teacher_logprobs" / f"example_{index:04d}.npz") as value:
            teacher = value["sample_logprob"].astype(np.float32)
        gap = student - teacher
        response_ids = np.asarray(row["response_token_ids"], dtype=np.int32)
        if gap.shape != response_ids.shape or not np.isfinite(gap).all():
            raise RuntimeError(f"index={index}: invalid score shape or value")
        matches = np.flatnonzero(response_ids == think_end_id)
        if len(matches) != 1:
            raise RuntimeError(f"index={index}: expected one </think>, found {len(matches)}")
        close = int(matches[0])
        delimiter_count += 1
        total_scored += len(gap)
        group = "correct" if bool(row["correct"]) else "wrong"

        whole_bins = np.minimum(9, np.floor(np.arange(len(gap)) * 10 / max(1, len(gap))).astype(np.int8))
        for decile in range(10):
            values = gap[whole_bins == decile]
            buckets["whole_deciles"][f"all:{decile}"].append(values)
            buckets["whole_deciles"][f"{group}:{decile}"].append(values)

        reasoning = gap[:close]
        final = gap[close + 1 :]
        if len(reasoning) == 0 or len(final) == 0:
            raise RuntimeError(f"index={index}: empty reasoning or final region")
        reasoning_bins = np.minimum(
            9, np.floor(np.arange(len(reasoning)) * 10 / len(reasoning)).astype(np.int8)
        )
        for decile in range(10):
            values = reasoning[reasoning_bins == decile]
            buckets["reasoning_deciles"][f"all:{decile}"].append(values)
            buckets["reasoning_deciles"][f"{group}:{decile}"].append(values)

        phase_values = {
            "reasoning_first10": reasoning[reasoning_bins == 0],
            "reasoning_middle80": reasoning[(reasoning_bins >= 1) & (reasoning_bins <= 8)],
            "reasoning_last10": reasoning[reasoning_bins == 9],
            "final_answer": final,
        }
        trace_item: dict[str, Any] = {
            "dataset_index": index,
            "correct": bool(row["correct"]),
            "difficulty": float(row["difficulty"]),
            "reasoning_tokens": int(len(reasoning)),
            "final_tokens": int(len(final)),
        }
        for phase, values in phase_values.items():
            buckets["phases"][f"all:{phase}"].append(values)
            buckets["phases"][f"{group}:{phase}"].append(values)
            trace_item[f"{phase}_mean_gap"] = float(values.mean())
            trace_item[f"{phase}_mean_abs_gap"] = float(np.abs(values).mean())
            trace_item[f"{phase}_positive_fraction"] = float((values > 0).mean())
        trace_metrics.append(trace_item)

    aggregate: dict[str, Any] = {}
    for section, values_by_key in buckets.items():
        aggregate[section] = {
            key: metrics(parts) for key, parts in sorted(values_by_key.items())
        }

    rng = np.random.default_rng(20260822)
    bootstrap: dict[str, Any] = {}
    for phase in ("reasoning_first10", "reasoning_middle80", "reasoning_last10", "final_answer"):
        bootstrap[phase] = {}
        for metric_name in ("mean_gap", "mean_abs_gap", "positive_fraction"):
            key = f"{phase}_{metric_name}"
            correct = np.asarray([item[key] for item in trace_metrics if item["correct"]], dtype=np.float64)
            wrong = np.asarray([item[key] for item in trace_metrics if not item["correct"]], dtype=np.float64)
            bootstrap[phase][metric_name] = bootstrap_difference(correct, wrong, rng)

    phase_order = ("reasoning_first10", "reasoning_middle80", "reasoning_last10", "final_answer")
    phase_all = {phase: aggregate["phases"][f"all:{phase}"] for phase in phase_order}
    summary = {
        "status": "DEEPMATH_BASELINE_GAP_POSITION_ANALYSIS_OK",
        "definition": "g_t = log pi_student(y_t|s_t) - log pi_teacher(y_t|s_t)",
        "rollouts": len(rows),
        "scored_tokens": total_scored,
        "think_delimiters": delimiter_count,
        "delimiter_policy": "</think> is excluded from reasoning/final phase comparison but retained in whole-response deciles",
        "phase_order": list(phase_order),
        "peak_phase_by_mean_abs_gap": max(phase_order, key=lambda phase: phase_all[phase]["mean_abs_gap"]),
        "late_minus_early_mean_abs_gap": (
            phase_all["reasoning_last10"]["mean_abs_gap"]
            - phase_all["reasoning_first10"]["mean_abs_gap"]
        ),
        "final_minus_reasoning_last10_mean_abs_gap": (
            phase_all["final_answer"]["mean_abs_gap"]
            - phase_all["reasoning_last10"]["mean_abs_gap"]
        ),
    }
    output = {
        "summary": summary,
        "aggregate": aggregate,
        "bootstrap_correct_minus_wrong": bootstrap,
        "per_rollout": trace_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "phases": phase_all}, ensure_ascii=False))


if __name__ == "__main__":
    main()
