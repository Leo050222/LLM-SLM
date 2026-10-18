#!/usr/bin/env python3
"""Compare an API judge with cached candidate-level reference judgments."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR.parent))

from anchor_selector.core import atomic_write_json, atomic_write_jsonl  # noqa: E402
from anchor_selector.openlux_client import (  # noqa: E402
    OpenLuxClient,
    build_user_prompt,
    load_env_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--secrets", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def probability_log(value: float) -> float:
    if value <= 0.0:
        return float("-inf")
    return math.log(value)


def build_prompt(row: dict[str, Any]) -> str:
    student_logprob = probability_log(float(row["student_probability"]))
    teacher_logprob = probability_log(
        float(row["teacher_probability_on_student_token"])
    )
    candidate = {
        "response_position": int(row["response_position"]),
        "student_token_id": -1,
        "student_token_piece": row["student_token_piece"],
        "student_token_text": row["student_token_text"],
        "student_logprob": student_logprob,
        "student_probability": float(row["student_probability"]),
        "teacher_logprob_on_student_token": teacher_logprob,
        "teacher_probability_on_student_token": float(
            row["teacher_probability_on_student_token"]
        ),
        "signed_logprob_gap_student_minus_teacher": student_logprob
        - teacher_logprob,
        "absolute_logprob_gap": float(row["absolute_logprob_gap"]),
        "teacher_top_tokens": row["teacher_top_tokens"],
    }
    return build_user_prompt(
        {"problem": row["problem"]},
        candidate,
        str(row["prefix_tail"]),
        bool(row["prefix_truncated"]),
        int(row["prefix_original_chars"]),
    )


def client_from_config(config: dict[str, Any], key: str) -> OpenLuxClient:
    return OpenLuxClient(
        api_url=str(config["api_url"]),
        api_key=key,
        model=str(config["judge_model"]),
        temperature=float(config["judge_temperature"]),
        max_tokens=int(config["judge_max_tokens"]),
        timeout_seconds=float(config["judge_timeout_seconds"]),
        max_retries=int(config["judge_max_retries"]),
        reasoning_effort=config.get("judge_reasoning_effort"),
        extra_body=config.get("judge_extra_body"),
    )


def evaluate(
    row: dict[str, Any], config: dict[str, Any], key: str
) -> dict[str, Any]:
    started = time.perf_counter()
    judgment, api = client_from_config(config, key).judge(build_prompt(row))
    reference = row["reference"]
    return {
        "dataset_index": int(row["dataset_index"]),
        "response_position": int(row["response_position"]),
        "candidate_index": int(row["candidate_index"]),
        "absolute_logprob_gap": float(row["absolute_logprob_gap"]),
        "reference": reference,
        "judgment": judgment,
        "agree_select_anchor": bool(reference["select_anchor"])
        == bool(judgment["select_anchor"]),
        "agree_label": str(reference["label"]) == str(judgment["label"]),
        "latency_seconds": time.perf_counter() - started,
        "api": api,
    }


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def usage_sum(results: list[dict[str, Any]], name: str) -> int:
    total = 0
    for row in results:
        usage = row["api"].get("usage") or {}
        if name == "reasoning_tokens":
            details = usage.get("completion_tokens_details") or {}
            value = details.get(name, 0)
        else:
            value = usage.get(name, 0)
        if isinstance(value, (int, float)):
            total += int(value)
    return total


def summarize(
    requested: int,
    results: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    elapsed: float,
) -> dict[str, Any]:
    tp = sum(
        row["reference"]["select_anchor"] and row["judgment"]["select_anchor"]
        for row in results
    )
    tn = sum(
        not row["reference"]["select_anchor"]
        and not row["judgment"]["select_anchor"]
        for row in results
    )
    fp = sum(
        not row["reference"]["select_anchor"] and row["judgment"]["select_anchor"]
        for row in results
    )
    fn = sum(
        row["reference"]["select_anchor"] and not row["judgment"]["select_anchor"]
        for row in results
    )
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    latencies = [float(row["latency_seconds"]) for row in results]
    usage = {
        name: usage_sum(results, name)
        for name in (
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "total_tokens",
        )
    }
    return {
        "status": "CALIBRATION_COMPLETE" if not errors else "CALIBRATION_INCOMPLETE",
        "requested_samples": requested,
        "completed_samples": len(results),
        "failed_samples": len(errors),
        "reference_positive": sum(
            row["reference"]["select_anchor"] for row in results
        ),
        "judge_positive": sum(row["judgment"]["select_anchor"] for row in results),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "select_anchor_accuracy": ratio(tp + tn, len(results)),
        "select_anchor_precision": precision,
        "select_anchor_recall": recall,
        "select_anchor_f1": f1,
        "label_accuracy": ratio(
            sum(row["agree_label"] for row in results), len(results)
        ),
        "reference_labels": dict(
            Counter(row["reference"]["label"] for row in results)
        ),
        "judge_labels": dict(
            Counter(row["judgment"]["label"] for row in results)
        ),
        "latency_seconds_mean": statistics.fmean(latencies) if latencies else None,
        "latency_seconds_median": statistics.median(latencies) if latencies else None,
        "wall_seconds": elapsed,
        "samples_per_second": ratio(len(results), elapsed),
        "usage": usage,
        "response_models": dict(
            Counter(str(row["api"].get("response_model")) for row in results)
        ),
    }


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    rows = read_jsonl(args.evidence)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    key = load_env_file(args.secrets)["OPENLUX_API_KEY"]
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(evaluate, row, config, key): row for row in rows
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                errors.append(
                    {
                        "dataset_index": row["dataset_index"],
                        "response_position": row["response_position"],
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
    results.sort(key=lambda row: (row["dataset_index"], row["response_position"]))
    errors.sort(key=lambda row: (row["dataset_index"], row["response_position"]))
    elapsed = time.perf_counter() - started
    summary = summarize(len(rows), results, errors, elapsed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(args.output_dir / "comparisons.jsonl", results)
    atomic_write_jsonl(args.output_dir / "errors.jsonl", errors)
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
