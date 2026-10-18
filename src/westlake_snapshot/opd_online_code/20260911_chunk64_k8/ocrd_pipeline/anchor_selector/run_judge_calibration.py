#!/usr/bin/env python3
"""Compare a local judge with cached Luna candidate-level judgments."""

from __future__ import annotations

import argparse
import json
import random
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

from anchor_selector.core import atomic_write_json, atomic_write_jsonl
from anchor_selector.openlux_client import build_user_prompt
from anchor_selector.run_anchor_judge_batch import Runtime, load_api_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--reference-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--secrets", type=Path, default=Path("/nonexistent"))
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--mode", choices=("balanced", "natural"), default="balanced")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--workers", type=int, default=32)
    return parser.parse_args()


def reference_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("traces/trace_*/judgments.jsonl")):
        dataset_index = int(path.parent.name.removeprefix("trace_"))
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append({"dataset_index": dataset_index, **json.loads(line)})
    if not rows:
        raise FileNotFoundError(f"No cached judgments beneath {root}")
    return rows


def choose_rows(
    rows: list[dict[str, Any]], samples: int, mode: str, seed: int
) -> list[dict[str, Any]]:
    if samples <= 0:
        raise ValueError("--samples must be positive")
    rng = random.Random(seed)
    if mode == "natural":
        chosen = rng.sample(rows, min(samples, len(rows)))
    else:
        positive = [row for row in rows if row["judgment"]["select_anchor"]]
        negative = [row for row in rows if not row["judgment"]["select_anchor"]]
        positive_n = min(samples // 2, len(positive))
        negative_n = min(samples - positive_n, len(negative))
        if positive_n + negative_n < samples:
            positive_n = min(samples - negative_n, len(positive))
        chosen = rng.sample(positive, positive_n) + rng.sample(negative, negative_n)
        rng.shuffle(chosen)
    return chosen


def candidate_record_path(root: Path, dataset_index: int) -> Path:
    direct = (
        root
        / "shards"
        / f"shard_{dataset_index % 16:02d}"
        / "candidate_records"
        / f"trace_{dataset_index:05d}.json"
    )
    if direct.is_file():
        return direct
    matches = list(root.glob(f"shards/shard_*/candidate_records/trace_{dataset_index:05d}.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one candidate record for dataset_index={dataset_index}, got {matches}"
        )
    return matches[0]


def evaluate_one(row: dict[str, Any], input_root: Path, runtime: Runtime) -> dict[str, Any]:
    started = time.perf_counter()
    dataset_index = int(row["dataset_index"])
    record = json.loads(candidate_record_path(input_root, dataset_index).read_text(encoding="utf-8"))
    position = int(row["response_position"])
    candidate = next(
        value for value in record["candidates"] if int(value["response_position"]) == position
    )
    prefix, truncated, original_chars = runtime.prefix(record["trace"], position)
    prompt = build_user_prompt(record["trace"], candidate, prefix, truncated, original_chars)
    local, api = runtime.client().judge(prompt)
    return {
        "dataset_index": dataset_index,
        "response_position": position,
        "candidate_index": int(candidate["candidate_index"]),
        "absolute_logprob_gap": float(candidate["absolute_logprob_gap"]),
        "reference": row["judgment"],
        "local": local,
        "agree_select_anchor": bool(local["select_anchor"])
        == bool(row["judgment"]["select_anchor"]),
        "agree_label": str(local["label"]) == str(row["judgment"]["label"]),
        "latency_seconds": time.perf_counter() - started,
        "api": api,
    }


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(
    selected: list[dict[str, Any]], results: list[dict[str, Any]], errors: list[dict[str, Any]],
    mode: str, seed: int, elapsed: float
) -> dict[str, Any]:
    tp = sum(r["reference"]["select_anchor"] and r["local"]["select_anchor"] for r in results)
    tn = sum(not r["reference"]["select_anchor"] and not r["local"]["select_anchor"] for r in results)
    fp = sum(not r["reference"]["select_anchor"] and r["local"]["select_anchor"] for r in results)
    fn = sum(r["reference"]["select_anchor"] and not r["local"]["select_anchor"] for r in results)
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    latencies = [float(row["latency_seconds"]) for row in results]
    return {
        "status": "CALIBRATION_COMPLETE" if not errors else "CALIBRATION_INCOMPLETE",
        "mode": mode,
        "seed": seed,
        "requested_samples": len(selected),
        "completed_samples": len(results),
        "failed_samples": len(errors),
        "reference_positive": sum(row["reference"]["select_anchor"] for row in results),
        "local_positive": sum(row["local"]["select_anchor"] for row in results),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "select_anchor_accuracy": ratio(tp + tn, len(results)),
        "select_anchor_precision": precision,
        "select_anchor_recall": recall,
        "select_anchor_f1": f1,
        "label_accuracy": ratio(sum(row["agree_label"] for row in results), len(results)),
        "reference_labels": dict(Counter(row["reference"]["label"] for row in results)),
        "local_labels": dict(Counter(row["local"]["label"] for row in results)),
        "latency_seconds_mean": statistics.fmean(latencies) if latencies else None,
        "latency_seconds_median": statistics.median(latencies) if latencies else None,
        "wall_seconds": elapsed,
        "samples_per_second": ratio(len(results), elapsed),
    }


def main() -> None:
    from transformers import AutoTokenizer

    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = reference_rows(args.reference_output)
    selected = choose_rows(rows, args.samples, args.mode, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    runtime = Runtime(
        config,
        load_api_key(config, args.secrets),
        tokenizer,
        int(config["prefix_max_chars"]),
    )
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(evaluate_one, row, args.input_root, runtime): row for row in selected
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
    summary = summarize(
        selected, results, errors, args.mode, args.seed, time.perf_counter() - started
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(args.output_dir / "comparisons.jsonl", results)
    atomic_write_jsonl(args.output_dir / "errors.jsonl", errors)
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
