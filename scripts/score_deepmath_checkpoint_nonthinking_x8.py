#!/usr/bin/env python3
"""Verify complete rollout shards, then score them on one CPU node."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from run_opd_logit_diagnostic import score_math_response, write_json, write_jsonl


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score_one(payload: tuple[int, dict[str, Any], float]) -> tuple[int, bool, str | None, str]:
    index, row, timeout = payload
    correct, predicted, method = score_math_response(
        str(row["response"]), str(row["solution"]), timeout
    )
    return index, bool(correct), predicted, method


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--score-timeout", type=float, default=20.0)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {args.output_dir}")

    rows: list[dict[str, Any]] = []
    shard_hashes: dict[str, str] = {}
    generation_seconds = 0.0
    for shard_index in range(args.num_shards):
        shard = args.shard_root / f"shard_{shard_index}"
        if not (shard / "SHARD_COMPLETE").is_file():
            raise RuntimeError(f"Incomplete shard: {shard}")
        config = json.loads((shard / "run_config.json").read_text())
        if config.get("student") != args.expected_model:
            raise RuntimeError(f"Model mismatch: {shard}")
        if int(config.get("shard_index", -1)) != shard_index:
            raise RuntimeError(f"Shard index mismatch: {shard}")
        path = shard / "rollouts.jsonl"
        observed_hash = sha256(path)
        if observed_hash != config.get("output_sha256"):
            raise RuntimeError(f"SHA mismatch: {path}")
        shard_rows = read_jsonl(path)
        expected = (1000 // args.num_shards) * args.num_rollouts
        if len(shard_rows) != expected:
            raise RuntimeError(f"Count mismatch: {path}")
        if any(int(row["dataset_index"]) % args.num_shards != shard_index for row in shard_rows):
            raise RuntimeError(f"Membership mismatch: {path}")
        rows.extend(shard_rows)
        shard_hashes[str(shard_index)] = observed_hash
        generation_seconds += float(config["elapsed_seconds_generate"])

    rows.sort(key=lambda row: (int(row["dataset_index"]), int(row["replicate_index"])))
    if len(rows) != 1000 * args.num_rollouts:
        raise RuntimeError("Total rollout count mismatch")
    keys = [(int(row["dataset_index"]), int(row["replicate_index"])) for row in rows]
    expected_keys = [(i, j) for i in range(1000) for j in range(args.num_rollouts)]
    if keys != expected_keys:
        raise RuntimeError("Dataset/replicate coverage mismatch")

    payloads = [(index, row, args.score_timeout) for index, row in enumerate(rows)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, correct, predicted, method in pool.map(score_one, payloads):
            rows[index]["correct"] = correct
            rows[index]["predicted_answer"] = predicted
            rows[index]["score_method"] = method

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["dataset_index"])].append(row)
    histogram = Counter(sum(bool(row["correct"]) for row in group) for group in grouped.values())
    correct = sum(bool(row["correct"]) for row in rows)
    lengths = [int(row["response_tokens"]) for row in rows]
    open_tags = sum(str(row["response"]).count("<think>") for row in rows)
    close_tags = sum(str(row["response"]).count("</think>") for row in rows)
    summary = {
        "status": "DEEPMATH_CHECKPOINT_NONTHINKING_X8_SCORING_COMPLETE",
        "model_label": args.model_label,
        "model": args.expected_model,
        "questions": 1000,
        "rollouts_per_question": args.num_rollouts,
        "rollouts": len(rows),
        "correct_rollouts": correct,
        "accuracy": correct / len(rows),
        "pass_at_8_count": 1000 - histogram[0],
        "pass_at_8": (1000 - histogram[0]) / 1000,
        "all_8_correct_count": histogram[8],
        "all_8_correct_rate": histogram[8] / 1000,
        "correct_count_histogram": {str(i): histogram[i] for i in range(9)},
        "mean_response_tokens": sum(lengths) / len(lengths),
        "total_response_tokens": sum(lengths),
        "max_response_tokens_observed": max(lengths),
        "thinking_open_tag": open_tags,
        "thinking_close_tag": close_tags,
        "generation_elapsed_seconds_sum_shards": generation_seconds,
        "score_workers": args.workers,
        "score_timeout_seconds": args.score_timeout,
        "source_shard_sha256": shard_hashes,
    }
    args.output_dir.mkdir(parents=True)
    write_jsonl(args.output_dir / "rollouts_scored.jsonl", rows)
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "SCORING_COMPLETE").touch()
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    print(summary["status"])


if __name__ == "__main__":
    main()
