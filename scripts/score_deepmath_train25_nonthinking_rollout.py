#!/usr/bin/env python3
"""Rule-score one immutable shard of saved DeepMath rollout traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from run_opd_logit_diagnostic import score_math_response


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--expected-rows", type=int, default=888)
    parser.add_argument("--score-timeout", type=float, default=20.0)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    rows = read_jsonl(args.rollouts)
    if len(rows) != args.expected_rows:
        raise RuntimeError(f"Expected {args.expected_rows} rows, got {len(rows)}")
    if any(int(row["dataset_index"]) % args.num_shards != args.shard_index for row in rows):
        raise RuntimeError("Modulo-shard membership mismatch")

    started = time.time()
    scores: list[dict[str, Any]] = []
    for offset, row in enumerate(rows, start=1):
        correct, predicted, method = score_math_response(
            str(row["response"]), str(row["solution"]), args.score_timeout
        )
        scores.append(
            {
                "dataset_index": int(row["dataset_index"]),
                "problem_id": str(row["problem_id"]),
                "correct": bool(correct),
                "predicted_answer": predicted,
                "target_answer": str(row["solution"]),
                "score_method": str(method),
            }
        )
        if offset % 50 == 0 or offset == len(rows):
            print(
                f"SCORE_PROGRESS shard={args.shard_index} rows={offset}/{len(rows)} "
                f"correct={sum(item['correct'] for item in scores)}",
                flush=True,
            )

    args.output_dir.mkdir(parents=True)
    score_path = args.output_dir / "scores.jsonl"
    atomic_jsonl(score_path, scores)
    config = {
        "status": "DEEPMATH_TRAIN25_NATIVE_NONTHINKING_SCORE_SHARD_OK",
        "rollouts": str(args.rollouts),
        "rollouts_sha256": sha256(args.rollouts),
        "scores_sha256": sha256(score_path),
        "scorer": "run_opd_logit_diagnostic.score_math_response",
        "score_timeout_seconds": args.score_timeout,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "rows": len(scores),
        "correct": sum(item["correct"] for item in scores),
        "score_methods": dict(sorted(Counter(item["score_method"] for item in scores).items())),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(args.output_dir / "score_config.json", config)
    (args.output_dir / "SCORE_SHARD_COMPLETE").touch()
    print(json.dumps(config, ensure_ascii=False, sort_keys=True))
    print("DEEPMATH_TRAIN25_NATIVE_NONTHINKING_SCORE_SHARD_OK")


if __name__ == "__main__":
    main()
