#!/usr/bin/env python3
"""Strictly verify a complete full-rollout batch and write its manifest."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

MODULE_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = MODULE_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from full_rollout.run_full_rollout_batch import (  # noqa: E402
    MODES,
    STUDENT_MODE,
    STUDENT_PREFIX_MODE,
    TEACHER_MODE,
    bundle_status,
    discover_records,
    output_filename,
    output_path,
    prompt_token_count,
    shard_records,
)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def expected_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    discovery_args = SimpleNamespace(
        mode=args.mode,
        anchor_root=args.anchor_root,
        bridge_manifest=args.bridge_manifest,
    )
    all_rows = discover_records(discovery_args)
    rows: list[dict[str, Any]] = []
    for shard_id in range(args.num_shards):
        rows.extend(
            shard_records(
                all_rows, shard_id, args.num_shards, args.max_records
            )
        )
    rows.sort(key=lambda row: int(row["anchor"]["dataset_index"]))
    return rows


def is_token_id(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def is_token_id_list(value: Any, *, nonempty: bool = False) -> bool:
    return (
        isinstance(value, list)
        and (not nonempty or bool(value))
        and all(is_token_id(token_id) for token_id in value)
    )


def normalized_token_id_set(value: Any) -> set[int]:
    """Normalize an int or iterable of ints into a strict token-id set."""
    if is_token_id(value):
        return {int(value)}
    if isinstance(value, (list, tuple, set)):
        token_ids = {int(token_id) for token_id in value if is_token_id(token_id)}
        if len(token_ids) != len(value):
            raise ValueError(f"invalid special token ids: {value!r}")
        return token_ids
    if value is None:
        return set()
    raise ValueError(f"invalid special token ids: {value!r}")


def has_nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def validate_bundle(
    value: dict[str, Any],
    row: dict[str, Any],
    args: argparse.Namespace,
    tokenizer: Any,
) -> tuple[int, dict[str, int], int]:
    anchor = row["anchor"]
    index = int(anchor["dataset_index"])
    prefix_ids = [int(token_id) for token_id in anchor["anchor_prefix_token_ids"]]
    prompt_count = prompt_token_count(anchor)
    response_prefix = prefix_ids[prompt_count:]
    rollouts = value.get("rollouts")
    expected_sampling = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "base_seed": args.seed,
        "stop_at_eos": True,
    }
    if not bool(
        value.get("status") == bundle_status(args.mode)
        and value.get("schema_version") == 1
        and value.get("mode") == args.mode
        and value.get("dataset_index") == index
        and value.get("problem_id") == anchor.get("problem_id")
        and value.get("trace_sha256") == anchor.get("trace_sha256")
        and value.get("anchor_artifact") == str(row["anchor_path"])
        and value.get("model") == args.model
        and value.get("tokenizer") == args.tokenizer
        and value.get("max_model_len") == args.max_model_len
        and value.get("requested_rollouts") == args.num_rollouts
        and value.get("actual_rollouts") == args.num_rollouts
        and value.get("prompt_token_count") == prompt_count
        and value.get("prompt_text")
        == tokenizer.decode(prefix_ids[:prompt_count], skip_special_tokens=False)
        and value.get("response_prefix_before_anchor_token_ids")
        == response_prefix
        and is_token_id_list(value.get("response_prefix_before_anchor_token_ids"))
        and value.get("response_prefix_before_anchor_text")
        == tokenizer.decode(response_prefix, skip_special_tokens=False)
        and value.get("student_anchor_token_id")
        == int(anchor["student_candidate_token_id"])
        and value.get("student_anchor_token_text")
        == anchor.get("student_candidate_token_text")
        and value.get("sampling") == expected_sampling
        and (
            args.mode != STUDENT_MODE
            or value.get("source_bridge_token_limit") == args.bridge_token_limit
        )
        and isinstance(rollouts, list)
        and len(rollouts) == args.num_rollouts
    ):
        raise ValueError(f"trace {index}: invalid bundle metadata")

    source_bridges = None
    if args.mode == STUDENT_MODE:
        if value.get("bridge_artifact") != str(row["bridge_path"]):
            raise ValueError(f"trace {index}: bridge artifact mismatch")
        source_bridges = row["bridge"].get("rollouts")
        if not isinstance(source_bridges, list) or len(source_bridges) != args.num_rollouts:
            raise ValueError(f"trace {index}: invalid source bridge bundle")

    token_total = 0
    finish_reasons: dict[str, int] = {}
    thinking_count = 0
    student_anchor = int(anchor["student_candidate_token_id"])
    for rollout_index, rollout in enumerate(rollouts):
        if rollout.get("rollout_index") != rollout_index:
            raise ValueError(f"trace {index}: rollout index/order mismatch")
        if not isinstance(rollout.get("contains_thinking_tag"), bool):
            raise ValueError(f"trace {index}: invalid thinking-tag flag")
        finish = str(rollout.get("finish_reason"))
        if finish not in {"stop", "length"}:
            raise ValueError(f"trace {index}: unexpected finish reason {finish}")
        if "stop_reason" not in rollout:
            raise ValueError(f"trace {index}: missing stop reason")
        if rollout.get("reached_model_length") is not (finish == "length"):
            raise ValueError(f"trace {index}: reached-model-length mismatch")
        finish_reasons[finish] = finish_reasons.get(finish, 0) + 1
        thinking_count += int(bool(rollout.get("contains_thinking_tag")))

        if args.mode == TEACHER_MODE:
            generated = rollout.get("teacher_completion_token_ids")
            if not is_token_id_list(generated, nonempty=True):
                raise ValueError(f"trace {index}: empty teacher completion")
            expected_first_seed = args.seed + 2 * (
                index * args.num_rollouts + rollout_index
            )
            if (
                rollout.get("first_step_seed") != expected_first_seed
                or rollout.get("continuation_seed") != expected_first_seed + 1
            ):
                raise ValueError(f"trace {index}: teacher seed mismatch")
            if rollout.get("teacher_completion_text") != tokenizer.decode(
                generated, skip_special_tokens=False
            ):
                raise ValueError(f"trace {index}: teacher completion text mismatch")
            if generated[0] == student_anchor:
                raise ValueError(f"trace {index}: teacher first token equals student")
            if rollout.get("teacher_completion_token_count") != len(generated):
                raise ValueError(f"trace {index}: teacher token count mismatch")
            first_step = rollout.get("first_step", {})
            if not isinstance(first_step, dict):
                raise ValueError(f"trace {index}: invalid first-step metadata")
            if first_step.get("chosen_token_id") != generated[0]:
                raise ValueError(f"trace {index}: first-step token mismatch")
            if first_step.get("excluded_student_token_id") != student_anchor:
                raise ValueError(f"trace {index}: excluded token mismatch")
            if first_step.get("chosen_token_text") != tokenizer.decode(
                [generated[0]], skip_special_tokens=False
            ):
                raise ValueError(f"trace {index}: chosen-token text mismatch")
            chosen_logprob = first_step.get("chosen_logprob")
            chosen_probability = first_step.get("chosen_probability")
            if not (
                isinstance(chosen_logprob, (int, float))
                and not isinstance(chosen_logprob, bool)
                and math.isfinite(chosen_logprob)
                and isinstance(chosen_probability, (int, float))
                and not isinstance(chosen_probability, bool)
                and math.isclose(
                    chosen_probability,
                    math.exp(chosen_logprob),
                    rel_tol=1e-12,
                    abs_tol=0.0,
                )
            ):
                raise ValueError(f"trace {index}: invalid chosen-token logprob")
            top_logprobs = first_step.get(
                "top_logprobs_after_first_token_constraint"
            )
            if not isinstance(top_logprobs, list) or len(top_logprobs) != 16:
                raise ValueError(f"trace {index}: missing first-step logprobs")
            seen_top_ids: set[int] = set()
            previous_logprob = math.inf
            for rank, item in enumerate(top_logprobs, start=1):
                if not isinstance(item, dict) or item.get("rank") != rank:
                    raise ValueError(f"trace {index}: invalid top-logprob rank")
                token_id = item.get("token_id")
                logprob = item.get("logprob")
                probability = item.get("probability")
                if not (
                    is_token_id(token_id)
                    and token_id not in seen_top_ids
                    and item.get("token_text")
                    == tokenizer.decode([token_id], skip_special_tokens=False)
                    and isinstance(logprob, (int, float))
                    and not isinstance(logprob, bool)
                    and math.isfinite(logprob)
                    and logprob <= previous_logprob
                    and isinstance(probability, (int, float))
                    and not isinstance(probability, bool)
                    and math.isclose(
                        probability,
                        math.exp(logprob) if logprob > -745.0 else 0.0,
                        rel_tol=1e-12,
                        abs_tol=0.0,
                    )
                ):
                    raise ValueError(f"trace {index}: invalid top-logprob entry")
                seen_top_ids.add(token_id)
                previous_logprob = logprob
            for item in top_logprobs:
                if item["token_id"] == generated[0] and not math.isclose(
                    item["logprob"], chosen_logprob, rel_tol=1e-12, abs_tol=0.0
                ):
                    raise ValueError(f"trace {index}: chosen logprob mismatch")
            expected_full_trace = [*response_prefix, *generated]
            if len(prefix_ids) + len(generated) > args.max_model_len:
                raise ValueError(f"trace {index}: teacher sequence exceeds context")
        elif args.mode == STUDENT_MODE:
            continuation = rollout.get("student_continuation_token_ids")
            bridge_ids = rollout.get("source_bridge_token_ids")
            if not is_token_id_list(continuation) or not is_token_id_list(
                bridge_ids, nonempty=True
            ):
                raise ValueError(f"trace {index}: invalid student continuation")
            expected_seed = args.seed + index * args.num_rollouts + rollout_index
            if rollout.get("student_seed") != expected_seed:
                raise ValueError(f"trace {index}: student seed mismatch")
            if rollout.get("source_bridge_text") != tokenizer.decode(
                bridge_ids, skip_special_tokens=False
            ):
                raise ValueError(f"trace {index}: source bridge text mismatch")
            if rollout.get("student_continuation_text") != tokenizer.decode(
                continuation, skip_special_tokens=False
            ):
                raise ValueError(f"trace {index}: continuation text mismatch")
            full_expected_bridge = source_bridges[rollout_index]["bridge_token_ids"]
            expected_bridge = (
                full_expected_bridge[: args.bridge_token_limit]
                if args.bridge_token_limit is not None
                else full_expected_bridge
            )
            if bridge_ids != expected_bridge:
                raise ValueError(f"trace {index}: source bridge mismatch")
            if rollout.get("source_bridge_token_count") != len(bridge_ids):
                raise ValueError(f"trace {index}: bridge token count mismatch")
            if (
                args.bridge_token_limit is not None
                and rollout.get("source_bridge_original_token_count")
                != len(full_expected_bridge)
            ):
                raise ValueError(f"trace {index}: original bridge token count mismatch")
            if rollout.get("student_continuation_token_count") != len(continuation):
                raise ValueError(f"trace {index}: continuation token count mismatch")
            expected_full_trace = [*response_prefix, *bridge_ids, *continuation]
            if len(prefix_ids) + len(bridge_ids) + len(continuation) > args.max_model_len:
                raise ValueError(f"trace {index}: student sequence exceeds context")
            generated = continuation
        else:
            generated = rollout.get("student_completion_token_ids")
            if not is_token_id_list(generated, nonempty=True):
                raise ValueError(f"trace {index}: empty student completion")
            expected_seed = args.seed + index * args.num_rollouts + rollout_index
            if rollout.get("student_seed") != expected_seed:
                raise ValueError(f"trace {index}: student seed mismatch")
            if rollout.get("student_completion_text") != tokenizer.decode(
                generated, skip_special_tokens=False
            ):
                raise ValueError(f"trace {index}: student completion text mismatch")
            if rollout.get("student_completion_token_count") != len(generated):
                raise ValueError(f"trace {index}: student completion count mismatch")
            expected_first_match = generated[0] == student_anchor
            if rollout.get(
                "first_token_matches_original_student_anchor"
            ) is not expected_first_match:
                raise ValueError(f"trace {index}: first-token match flag mismatch")
            expected_full_trace = [*response_prefix, *generated]
            if len(prefix_ids) + len(generated) > args.max_model_len:
                raise ValueError(f"trace {index}: student sequence exceeds context")

        full_trace = rollout.get("full_trace_token_ids")
        if not is_token_id_list(full_trace, nonempty=True) or full_trace != expected_full_trace:
            raise ValueError(f"trace {index}: full trace token mismatch")
        if rollout.get("full_trace_token_count") != len(full_trace):
            raise ValueError(f"trace {index}: full trace count mismatch")
        if rollout.get("full_trace_text") != tokenizer.decode(
            full_trace, skip_special_tokens=False
        ):
            raise ValueError(f"trace {index}: full trace text mismatch")
        total_sequence_length = prompt_count + len(full_trace)
        expected_thinking_text = tokenizer.decode(
            generated, skip_special_tokens=False
        )
        if rollout.get("contains_thinking_tag") is not any(
            tag in expected_thinking_text for tag in ("<think>", "</think>")
        ):
            raise ValueError(f"trace {index}: thinking-tag flag mismatch")
        if finish == "length":
            if total_sequence_length != args.max_model_len or rollout.get(
                "stop_reason"
            ) is not None:
                raise ValueError(f"trace {index}: invalid length termination")
        else:
            eos_ids = set(args.accepted_eos_token_ids)
            if not generated or generated[-1] not in eos_ids:
                raise ValueError(f"trace {index}: stop does not end in EOS")
            stop_reason = rollout.get("stop_reason")
            if stop_reason is not None and stop_reason != generated[-1]:
                raise ValueError(f"trace {index}: stop reason does not match EOS")
        token_total += len(generated)

    if args.mode == TEACHER_MODE and value.get(
        "all_first_tokens_differ_from_student"
    ) is not True:
        raise ValueError(f"trace {index}: first-token aggregate flag false")
    if args.mode == STUDENT_PREFIX_MODE:
        expected_match_count = sum(
            bool(rollout["first_token_matches_original_student_anchor"])
            for rollout in rollouts
        )
        if value.get(
            "first_token_matches_original_student_anchor_count"
        ) != expected_match_count:
            raise ValueError(f"trace {index}: first-token aggregate count mismatch")
    return token_total, finish_reasons, thinking_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--anchor-root", type=Path)
    parser.add_argument("--bridge-manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument(
        "--bridge-token-limit",
        type=int,
        help="For student_after_bridge, require the first N bridge tokens only.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate every artifact without writing manifest/summary/markers",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.tokenizer = args.tokenizer or args.model
    if args.mode in {TEACHER_MODE, STUDENT_PREFIX_MODE} and args.anchor_root is None:
        raise ValueError("--anchor-root is required for anchor-prefix modes")
    if args.mode == STUDENT_MODE and args.bridge_manifest is None:
        raise ValueError("--bridge-manifest is required for student mode")
    if args.bridge_token_limit is not None:
        if args.mode != STUDENT_MODE:
            raise ValueError("--bridge-token-limit is only valid for student_after_bridge")
        if args.bridge_token_limit <= 0:
            raise ValueError("--bridge-token-limit must be positive")
    rows = expected_rows(args)
    from transformers import AutoTokenizer, GenerationConfig

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    generation_config = GenerationConfig.from_pretrained(
        args.model, local_files_only=True
    )
    args.accepted_eos_token_ids = sorted(
        normalized_token_id_set(tokenizer.eos_token_id)
        | normalized_token_id_set(generation_config.eos_token_id)
    )
    if not args.accepted_eos_token_ids:
        raise ValueError("model/tokenizer do not define any EOS token ids")
    if args.expected_count is not None and len(rows) != args.expected_count:
        raise ValueError(f"expected rows {len(rows)} != {args.expected_count}")
    for shard_id in range(args.num_shards):
        marker = (
            args.output_root
            / "shards"
            / f"shard_{shard_id:02d}"
            / "SHARD_COMPLETE"
        )
        if not marker.is_file():
            raise FileNotFoundError(f"missing shard marker: {marker}")

    expected_paths: set[Path] = set()
    manifest_rows: list[dict[str, Any]] = []
    total_tokens = 0
    finish_reasons: dict[str, int] = {}
    thinking_tag_rollouts = 0
    first_token_matches_original_anchor = 0
    for row in rows:
        index = int(row["anchor"]["dataset_index"])
        path = output_path(
            args.output_root, index, args.mode, args.num_rollouts
        )
        expected_paths.add(path.resolve())
        value = json.loads(path.read_text(encoding="utf-8"))
        tokens, finishes, thinking = validate_bundle(
            value, row, args, tokenizer
        )
        total_tokens += tokens
        thinking_tag_rollouts += thinking
        if args.mode == STUDENT_PREFIX_MODE:
            first_token_matches_original_anchor += int(
                value["first_token_matches_original_student_anchor_count"]
            )
        for reason, count in finishes.items():
            finish_reasons[reason] = finish_reasons.get(reason, 0) + count
        manifest_row = {
            "dataset_index": index,
            "problem_id": row["anchor"].get("problem_id"),
            "trace_sha256": row["anchor"].get("trace_sha256"),
            "anchor_artifact": str(row["anchor_path"]),
            "rollout_artifact": str(path),
            "rollout_count": args.num_rollouts,
            "mode": args.mode,
        }
        if args.mode == STUDENT_MODE:
            manifest_row["bridge_artifact"] = str(row["bridge_path"])
        manifest_rows.append(manifest_row)

    actual_paths = {
        path.resolve()
        for path in args.output_root.glob(
            f"traces/trace_*/{output_filename(args.mode, args.num_rollouts)}"
        )
    }
    missing = expected_paths - actual_paths
    extras = actual_paths - expected_paths
    if missing or extras:
        raise ValueError(
            f"artifact set mismatch: missing={len(missing)} extras={len(extras)}"
        )

    summary = {
        "status": "FULL_ROLLOUT_BATCH_CHECK_ONLY_OK"
        if args.check_only
        else "FULL_ROLLOUT_BATCH_COMPLETE",
        "mode": args.mode,
        "verified_trace_count": len(rows),
        "verified_rollout_count": len(rows) * args.num_rollouts,
        "verified_generated_token_count": total_tokens,
        "requested_rollouts_per_trace": args.num_rollouts,
        "model": args.model,
        "accepted_eos_token_ids": args.accepted_eos_token_ids,
        "max_model_len": args.max_model_len,
        "finish_reason_counts": finish_reasons,
        "thinking_tag_rollout_count": thinking_tag_rollouts,
        "first_token_differs_count": len(rows) * args.num_rollouts
        if args.mode == TEACHER_MODE
        else None,
        "first_token_matches_original_student_anchor_count":
        first_token_matches_original_anchor
        if args.mode == STUDENT_PREFIX_MODE
        else None,
        "num_shards": args.num_shards,
        "source_bridge_token_limit": args.bridge_token_limit
        if args.mode == STUDENT_MODE
        else None,
        "missing_artifact_count": 0,
        "extra_artifact_count": 0,
    }
    if args.check_only:
        print(json.dumps(summary, sort_keys=True), flush=True)
        return
    manifest_path = args.output_root / "manifest.jsonl"
    temporary = manifest_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    atomic_write_json(args.output_root / "summary.json", summary)
    (args.output_root / "FULL_ROLLOUT_BATCH_COMPLETE").touch()
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
