#!/usr/bin/env python3
"""Score one deterministic shard of teacher or student full rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROLLOUT_KINDS = (
    "teacher_full",
    "student_after_bridge",
    "student_from_anchor_prefix",
)


def shard_status(rollout_kind: str) -> str:
    if rollout_kind == "teacher_full":
        return "TEACHER_ROLLOUT_ANALYSIS_SHARD_COMPLETE"
    if rollout_kind == "student_after_bridge":
        return "STUDENT_AFTER_BRIDGE_ROLLOUT_ANALYSIS_SHARD_COMPLETE"
    return "STUDENT_FROM_ANCHOR_PREFIX_ROLLOUT_ANALYSIS_SHARD_COMPLETE"


def boxed_expressions(text: str) -> list[str]:
    results: list[str] = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        depth = 1
        cursor = match.end()
        start = cursor
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            results.append(text[start : cursor - 1])
    return results


def _equiv_worker(candidate: str, target: str, connection: Any) -> None:
    try:
        from verl.utils.reward_score.math_reward import is_equiv

        connection.send(bool(is_equiv(candidate, target)))
    except BaseException:
        connection.send(False)
    finally:
        connection.close()


def normalized_boolean(value: str) -> str | None:
    text = value.strip().lower()
    outer_text = re.fullmatch(r"\\text\{(.*)\}", text, flags=re.S)
    if outer_text:
        text = outer_text.group(1).strip()
    text = re.sub(r"^[\s$\\{}]+|[\s$\\{}.,;:!*]+$", "", text)
    match = re.match(r"^(yes|no|true|false)\b", text)
    if not match:
        return None
    return {"true": "yes", "false": "no"}.get(match.group(1), match.group(1))


def score_math_response(
    response: str, target: str, timeout_seconds: float
) -> tuple[bool, str | None, str]:
    """Match the frozen DeepMath scorer used for the original student traces."""
    from verl.utils.reward_score.math_reward import strip_string

    final_region = response.rsplit("</think>", 1)[-1].strip()
    boxes = boxed_expressions(final_region)
    candidate = boxes[-1].strip() if boxes else None
    if candidate is None:
        tail = final_region[-1200:]
        markers = list(
            re.finditer(
                r"(?is)(?:final\s+answer|answer\s+is|answer\s*:|"
                r"therefore[, ]+the\s+answer\s+is|thus[, ]+the\s+answer\s+is)"
                r"\s*[:=]?\s*",
                tail,
            )
        )
        if markers:
            segment = tail[markers[-1].end() :].strip()
            math_spans = re.findall(r"\$\$([^$]+)\$\$|\$([^$]+)\$", segment, flags=re.S)
            if math_spans:
                candidate = next(value for value in reversed(math_spans[-1]) if value)
            else:
                lines = [line.strip() for line in segment.splitlines() if line.strip()]
                candidate = lines[-1] if lines else None
    if candidate is None:
        return False, None, "missing"

    candidate = candidate.strip().rstrip(".* ")
    target_boolean = normalized_boolean(str(target))
    candidate_boolean = normalized_boolean(candidate)
    if target_boolean is not None and candidate_boolean is not None:
        return candidate_boolean == target_boolean, candidate, "boolean_normalized"
    try:
        if strip_string(candidate) == strip_string(str(target)):
            return True, candidate, "normalized_exact"
    except Exception:
        pass

    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_equiv_worker, args=(candidate, str(target), child), daemon=True
    )
    process.start()
    child.close()
    try:
        if parent.poll(timeout_seconds):
            return bool(parent.recv()), candidate, "symbolic"
        return False, candidate, "timeout"
    except (EOFError, OSError):
        return False, candidate, "error"
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--score-timeout", type=float, default=20.0)
    parser.add_argument(
        "--expected-dataset-rows",
        type=int,
        default=14_158,
        help="Strict row-count witness for the source parquet.",
    )
    parser.add_argument(
        "--rollout-kind", choices=ROLLOUT_KINDS, default="teacher_full"
    )
    parser.add_argument(
        "--fail-after-traces-for-test",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    if args.rollout_kind in {"teacher_full", "student_from_anchor_prefix"}:
        required_marker = "FULL_ROLLOUT_BATCH_COMPLETE"
    elif (args.rollout_root / "GPU_GENERATION_COMPLETE").is_file():
        required_marker = "GPU_GENERATION_COMPLETE"
    else:
        required_marker = "FULL_ROLLOUT_BATCH_COMPLETE"
    if not (args.rollout_root / required_marker).is_file():
        raise FileNotFoundError(
            f"{args.rollout_kind} required source marker is missing: {required_marker}"
        )

    dataset = pd.read_parquet(args.dataset)
    if len(dataset) != args.expected_dataset_rows:
        raise ValueError(
            f"expected {args.expected_dataset_rows} source rows, got {len(dataset)}"
        )

    shard_dir = args.output_root / "shards" / f"shard_{args.shard_index:03d}"
    marker = shard_dir / "SCORE_SHARD_COMPLETE"
    if marker.is_file():
        config = json.loads((shard_dir / "summary.json").read_text(encoding="utf-8"))
        if not bool(
            config.get("status") == shard_status(args.rollout_kind)
            and config.get("rollout_kind", "teacher_full") == args.rollout_kind
        ):
            raise ValueError("cached shard marker/summary mismatch")
        print(json.dumps(config, sort_keys=True), flush=True)
        return
    shard_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = shard_dir / "trace_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    bundle_names = {
        "teacher_full": "teacher_full_n8.json",
        "student_after_bridge": "student_after_bridge_n8.json",
        "student_from_anchor_prefix": "student_from_anchor_prefix_n8.json",
    }
    bundle_name = bundle_names[args.rollout_kind]
    trace_paths = sorted(
        path
        for path in args.rollout_root.glob(f"traces/trace_*/{bundle_name}")
        if int(path.parent.name.removeprefix("trace_")) % args.num_shards
        == args.shard_index
    )
    if not trace_paths:
        raise ValueError("empty analysis shard")

    started = time.time()
    scores: list[dict[str, Any]] = []
    methods: Counter[str] = Counter()
    cached_traces = 0
    scored_traces_now = 0
    for trace_offset, path in enumerate(trace_paths, start=1):
        bundle = json.loads(path.read_text(encoding="utf-8"))
        dataset_index = int(bundle["dataset_index"])
        source = dataset.iloc[dataset_index]
        extra = source["extra_info"]
        ground_truth = str(source["reward_model"]["ground_truth"])
        problem_id = str(extra["problem_id"])
        if problem_id != str(bundle["problem_id"]):
            raise ValueError(f"trace {dataset_index}: dataset alignment mismatch")
        rollouts = bundle.get("rollouts")
        if not isinstance(rollouts, list) or len(rollouts) != args.num_rollouts:
            raise ValueError(f"trace {dataset_index}: invalid rollout count")
        response_prefix_tokens = len(bundle["response_prefix_before_anchor_token_ids"])
        checkpoint_path = checkpoint_dir / f"trace_{dataset_index:05d}.json"
        trace_scores: list[dict[str, Any]]
        if checkpoint_path.is_file():
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            trace_scores = checkpoint.get("scores")
            if not bool(
                checkpoint.get("status")
                == f"{args.rollout_kind.upper()}_TRACE_SCORE_COMPLETE"
                and checkpoint.get("rollout_kind", "teacher_full")
                == args.rollout_kind
                and int(checkpoint.get("dataset_index", -1)) == dataset_index
                and isinstance(trace_scores, list)
                and len(trace_scores) == args.num_rollouts
                and sorted(int(row["rollout_index"]) for row in trace_scores)
                == list(range(args.num_rollouts))
                and all(int(row["dataset_index"]) == dataset_index for row in trace_scores)
                and all(str(row["target_answer"]) == ground_truth for row in trace_scores)
            ):
                raise ValueError(f"trace {dataset_index}: invalid score checkpoint")
            cached_traces += 1
        else:
            trace_scores = []
            for rollout in rollouts:
                correct, predicted, method = score_math_response(
                    str(rollout["full_trace_text"]), ground_truth, args.score_timeout
                )
                score_row = {
                        "dataset_index": dataset_index,
                        "problem_id": problem_id,
                        "rollout_index": int(rollout["rollout_index"]),
                        "correct": bool(correct),
                        "predicted_answer": predicted,
                        "target_answer": ground_truth,
                        "score_method": str(method),
                        "difficulty": float(extra["difficulty"]),
                        "topic": str(extra["topic"]),
                        "response_prefix_token_count": response_prefix_tokens,
                        "completion_token_count": int(
                            rollout[
                                {
                                    "teacher_full": "teacher_completion_token_count",
                                    "student_after_bridge": "student_continuation_token_count",
                                    "student_from_anchor_prefix": "student_completion_token_count",
                                }[args.rollout_kind]
                            ]
                        ),
                        "full_trace_token_count": int(
                            rollout["full_trace_token_count"]
                        ),
                        "finish_reason": str(rollout["finish_reason"]),
                        "reached_model_length": bool(
                            rollout["reached_model_length"]
                        ),
                    }
                if args.rollout_kind == "teacher_full":
                    score_row["teacher_completion_token_count"] = int(
                        rollout["teacher_completion_token_count"]
                    )
                elif args.rollout_kind == "student_after_bridge":
                    score_row["source_bridge_token_count"] = int(
                        rollout["source_bridge_token_count"]
                    )
                    score_row["student_continuation_token_count"] = int(
                        rollout["student_continuation_token_count"]
                    )
                else:
                    score_row["student_completion_token_count"] = int(
                        rollout["student_completion_token_count"]
                    )
                    score_row[
                        "first_token_matches_original_student_anchor"
                    ] = bool(
                        rollout["first_token_matches_original_student_anchor"]
                    )
                trace_scores.append(score_row)
            atomic_json(
                checkpoint_path,
                {
                    "status": f"{args.rollout_kind.upper()}_TRACE_SCORE_COMPLETE",
                    "rollout_kind": args.rollout_kind,
                    "dataset_index": dataset_index,
                    "problem_id": problem_id,
                    "num_rollouts": args.num_rollouts,
                    "scores": trace_scores,
                },
            )
            scored_traces_now += 1
            if (
                args.fail_after_traces_for_test is not None
                and scored_traces_now >= args.fail_after_traces_for_test
            ):
                raise RuntimeError(
                    "intentional checkpoint-resume smoke failure after "
                    f"{scored_traces_now} newly scored traces"
                )
        for row in trace_scores:
            methods[str(row["score_method"])] += 1
        scores.extend(trace_scores)
        if trace_offset % 25 == 0 or trace_offset == len(trace_paths):
            print(
                f"ANALYSIS_PROGRESS shard={args.shard_index} "
                f"traces={trace_offset}/{len(trace_paths)} scores={len(scores)} "
                f"cached_traces={cached_traces} scored_now={scored_traces_now}",
                flush=True,
            )

    expected = len(trace_paths) * args.num_rollouts
    if len(scores) != expected:
        raise RuntimeError("analysis score count mismatch")
    scores_path = shard_dir / "scores.jsonl"
    atomic_jsonl(scores_path, scores)
    summary = {
        "status": shard_status(args.rollout_kind),
        "rollout_kind": args.rollout_kind,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "trace_count": len(trace_paths),
        "cached_trace_count": cached_traces,
        "scored_trace_count_now": scored_traces_now,
        "rollout_count": len(scores),
        "correct_count": sum(row["correct"] for row in scores),
        "score_methods": dict(sorted(methods.items())),
        "score_timeout_seconds": args.score_timeout,
        "scores_sha256": sha256(scores_path),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(shard_dir / "summary.json", summary)
    marker.touch()
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
