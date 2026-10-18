#!/usr/bin/env python3
"""Generate resumable full rollouts from anchors or completed bridges."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

MODULE_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = MODULE_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from bridge_generator.core import atomic_write_json, load_selected_anchor  # noqa: E402
from bridge_generator.run_bridge_batch import (  # noqa: E402
    discover_selected_anchors,
)
from bridge_generator.run_bridge_generator import (  # noqa: E402
    chosen_logprob,
    serialize_top_logprobs,
)


TEACHER_MODE = "teacher_from_anchor"
STUDENT_MODE = "student_after_bridge"
STUDENT_PREFIX_MODE = "student_from_anchor_prefix"
MODES = (TEACHER_MODE, STUDENT_MODE, STUDENT_PREFIX_MODE)
DEFAULT_TEACHER = "/storage/yukaichengLab/share/model/Qwen3-14B"
DEFAULT_STUDENT = "/storage/yukaichengLab/share/model/Qwen3-1.7B"


def output_filename(mode: str, num_rollouts: int) -> str:
    if mode == TEACHER_MODE:
        return f"teacher_full_n{num_rollouts}.json"
    if mode == STUDENT_MODE:
        return f"student_after_bridge_n{num_rollouts}.json"
    if mode == STUDENT_PREFIX_MODE:
        return f"student_from_anchor_prefix_n{num_rollouts}.json"
    raise ValueError(f"Unknown mode: {mode}")


def output_path(
    output_root: Path, dataset_index: int, mode: str, num_rollouts: int
) -> Path:
    return (
        output_root
        / "traces"
        / f"trace_{dataset_index:05d}"
        / output_filename(mode, num_rollouts)
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def prompt_token_count(anchor: dict[str, Any]) -> int:
    absolute = int(anchor["anchor_absolute_sequence_position"])
    response = int(anchor["anchor_response_position"])
    count = absolute - response
    if count <= 0:
        raise ValueError(
            f"trace {anchor.get('dataset_index')}: invalid prompt token count {count}"
        )
    recorded = anchor.get("prompt_tokens")
    if isinstance(recorded, int) and recorded != count:
        raise ValueError(
            f"trace {anchor.get('dataset_index')}: prompt count mismatch "
            f"recorded={recorded} derived={count}"
        )
    return count


def read_bridge_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Manifest line {line_number} is not an object")
            rows.append(value)
    rows.sort(key=lambda row: int(row["dataset_index"]))
    return rows


def discover_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if args.mode in {TEACHER_MODE, STUDENT_PREFIX_MODE}:
        for anchor_path, anchor in discover_selected_anchors(args.anchor_root):
            rows.append({"anchor_path": anchor_path, "anchor": anchor})
    else:
        for manifest_row in read_bridge_manifest(args.bridge_manifest):
            anchor_path = Path(manifest_row["anchor_artifact"])
            bridge_path = Path(manifest_row["bridge_artifact"])
            rows.append(
                {
                    "anchor_path": anchor_path,
                    "anchor": load_selected_anchor(anchor_path),
                    "bridge_path": bridge_path,
                    "bridge": read_json(bridge_path),
                }
            )
    rows.sort(key=lambda row: int(row["anchor"]["dataset_index"]))
    return rows


def shard_records(
    rows: list[dict[str, Any]],
    shard_id: int,
    num_shards: int,
    max_records: int | None,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if int(row["anchor"]["dataset_index"]) % num_shards == shard_id
    ]
    if max_records is not None:
        selected = selected[:max_records]
    return selected


def bundle_status(mode: str) -> str:
    if mode == TEACHER_MODE:
        return "TEACHER_FULL_ROLLOUTS_GENERATED"
    if mode == STUDENT_MODE:
        return "STUDENT_AFTER_BRIDGE_ROLLOUTS_GENERATED"
    if mode == STUDENT_PREFIX_MODE:
        return "STUDENT_FROM_ANCHOR_PREFIX_ROLLOUTS_GENERATED"
    raise ValueError(f"Unknown mode: {mode}")


def valid_cached_bundle(
    path: Path,
    row: dict[str, Any],
    mode: str,
    model: str,
    num_rollouts: int,
    max_model_len: int,
    bridge_token_limit: int | None,
) -> bool:
    if not path.is_file():
        return False
    try:
        value = read_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    anchor = row["anchor"]
    rollouts = value.get("rollouts")
    sampling = value.get("sampling")
    if not bool(
        value.get("status") == bundle_status(mode)
        and value.get("schema_version") == 1
        and value.get("mode") == mode
        and value.get("dataset_index") == int(anchor["dataset_index"])
        and value.get("problem_id") == anchor.get("problem_id")
        and value.get("trace_sha256") == anchor.get("trace_sha256")
        and value.get("anchor_artifact") == str(row["anchor_path"])
        and value.get("model") == model
        and isinstance(value.get("tokenizer"), str)
        and value.get("requested_rollouts") == num_rollouts
        and value.get("actual_rollouts") == num_rollouts
        and value.get("max_model_len") == max_model_len
        and (
            mode != STUDENT_MODE
            or value.get("source_bridge_token_limit") == bridge_token_limit
        )
        and isinstance(sampling, dict)
        and sampling.get("stop_at_eos") is True
        and isinstance(rollouts, list)
        and len(rollouts) == num_rollouts
    ):
        return False
    for rollout_index, rollout in enumerate(rollouts):
        if rollout.get("rollout_index") != rollout_index:
            return False
        full_trace = rollout.get("full_trace_token_ids")
        if not isinstance(full_trace, list) or not full_trace or not all(
            isinstance(token_id, int)
            and not isinstance(token_id, bool)
            and token_id >= 0
            for token_id in full_trace
        ):
            return False
        if not isinstance(rollout.get("full_trace_text"), str) or not rollout.get(
            "full_trace_text"
        ):
            return False
        if rollout.get("finish_reason") not in {"stop", "length"}:
            return False
        if "stop_reason" not in rollout:
            return False
        if mode == TEACHER_MODE:
            completion = rollout.get("teacher_completion_token_ids")
            if not isinstance(completion, list) or not completion:
                return False
            if completion[0] == int(anchor["student_candidate_token_id"]):
                return False
        elif mode == STUDENT_MODE:
            continuation = rollout.get("student_continuation_token_ids")
            if not isinstance(continuation, list):
                return False
            bridge_ids = rollout.get("source_bridge_token_ids")
            if not isinstance(bridge_ids, list) or not bridge_ids:
                return False
            if bridge_token_limit is not None and len(bridge_ids) != bridge_token_limit:
                return False
        else:
            completion = rollout.get("student_completion_token_ids")
            if not isinstance(completion, list) or not completion:
                return False
    return True


def prepare_common(row: dict[str, Any], max_model_len: int) -> dict[str, Any]:
    anchor = row["anchor"]
    prefix_ids = [int(token_id) for token_id in anchor["anchor_prefix_token_ids"]]
    prompt_count = prompt_token_count(anchor)
    if not prompt_count <= len(prefix_ids) < max_model_len:
        raise ValueError(
            f"trace {anchor['dataset_index']}: invalid prefix/prompt length"
        )
    prepared = dict(row)
    prepared["prefix_ids"] = prefix_ids
    prepared["prompt_token_count"] = prompt_count
    prepared["response_prefix_ids"] = prefix_ids[prompt_count:]
    return prepared


def eos_token_ids(tokenizer: Any) -> set[int]:
    eos = tokenizer.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, int):
        return {int(eos)}
    return {int(token_id) for token_id in eos}


def teacher_process_batch(
    rows: list[dict[str, Any]],
    *,
    llm: Any,
    tokenizer: Any,
    args: argparse.Namespace,
) -> dict[int, list[dict[str, Any]]]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    variants: list[dict[str, Any]] = []
    first_prompts = []
    first_params = []
    for row in rows:
        anchor = row["anchor"]
        index = int(anchor["dataset_index"])
        student_token = int(anchor["student_candidate_token_id"])
        for rollout_index in range(args.num_rollouts):
            first_seed = args.seed + 2 * (index * args.num_rollouts + rollout_index)
            variant = {
                **row,
                "rollout_index": rollout_index,
                "first_seed": first_seed,
                "continuation_seed": first_seed + 1,
                "student_token": student_token,
            }
            variants.append(variant)
            first_prompts.append(TokensPrompt(prompt_token_ids=row["prefix_ids"]))
            first_params.append(
                SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    seed=first_seed,
                    max_tokens=1,
                    min_tokens=1,
                    ignore_eos=True,
                    logprobs=args.first_step_logprobs,
                    _bad_words_token_ids=[[student_token]],
                    detokenize=False,
                    skip_special_tokens=False,
                )
            )

    first_outputs = llm.generate(first_prompts, first_params, use_tqdm=False)
    if len(first_outputs) != len(variants):
        raise RuntimeError("Teacher first-step output count mismatch")

    eos_ids = eos_token_ids(tokenizer)
    first_rows: list[dict[str, Any]] = []
    continuation_prompts = []
    continuation_params = []
    continuation_variant_positions: list[int] = []
    for position, (variant, request) in enumerate(
        zip(variants, first_outputs, strict=True)
    ):
        generated = request.outputs[0]
        token_ids = [int(token_id) for token_id in generated.token_ids]
        if len(token_ids) != 1:
            raise RuntimeError("Teacher first step did not produce exactly one token")
        first_id = token_ids[0]
        if first_id == variant["student_token"]:
            raise RuntimeError(
                f"trace {variant['anchor']['dataset_index']}: first-token exclusion failed"
            )
        entry = generated.logprobs[0] if generated.logprobs else None
        first_rows.append(
            {
                "token_id": first_id,
                "token_text": tokenizer.decode(
                    [first_id], skip_special_tokens=False
                ),
                "entry": entry,
                "chosen_logprob": chosen_logprob(entry, first_id),
                "is_eos": first_id in eos_ids,
            }
        )
        if first_id not in eos_ids:
            max_tokens = args.max_model_len - len(variant["prefix_ids"]) - 1
            if max_tokens <= 0:
                raise ValueError("No room for teacher continuation")
            continuation_variant_positions.append(position)
            continuation_prompts.append(
                TokensPrompt(
                    prompt_token_ids=variant["prefix_ids"] + [first_id]
                )
            )
            continuation_params.append(
                SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    seed=variant["continuation_seed"],
                    max_tokens=max_tokens,
                    n=1,
                    detokenize=False,
                    skip_special_tokens=False,
                )
            )

    continuation_outputs = llm.generate(
        continuation_prompts, continuation_params, use_tqdm=False
    ) if continuation_prompts else []
    if len(continuation_outputs) != len(continuation_variant_positions):
        raise RuntimeError("Teacher continuation output count mismatch")
    continuation_by_position = dict(
        zip(continuation_variant_positions, continuation_outputs, strict=True)
    )

    grouped: dict[int, list[dict[str, Any]]] = {}
    for position, (variant, first) in enumerate(
        zip(variants, first_rows, strict=True)
    ):
        if first["is_eos"]:
            continuation_ids: list[int] = []
            finish_reason = "stop"
            stop_reason: Any = first["token_id"]
        else:
            generated = continuation_by_position[position].outputs[0]
            continuation_ids = [int(token_id) for token_id in generated.token_ids]
            finish_reason = generated.finish_reason
            stop_reason = generated.stop_reason
        completion_ids = [first["token_id"], *continuation_ids]
        full_trace_ids = [*variant["response_prefix_ids"], *completion_ids]
        index = int(variant["anchor"]["dataset_index"])
        first_logprob = first["chosen_logprob"]
        artifact = {
            "rollout_index": variant["rollout_index"],
            "first_step_seed": variant["first_seed"],
            "continuation_seed": variant["continuation_seed"],
            "teacher_completion_token_ids": completion_ids,
            "teacher_completion_text": tokenizer.decode(
                completion_ids, skip_special_tokens=False
            ),
            "teacher_completion_token_count": len(completion_ids),
            "full_trace_token_ids": full_trace_ids,
            "full_trace_text": tokenizer.decode(
                full_trace_ids, skip_special_tokens=False
            ),
            "full_trace_token_count": len(full_trace_ids),
            "finish_reason": finish_reason,
            "stop_reason": stop_reason,
            "reached_model_length": finish_reason == "length",
            "contains_thinking_tag": any(
                tag in tokenizer.decode(completion_ids, skip_special_tokens=False)
                for tag in ("<think>", "</think>")
            ),
            "first_step": {
                "excluded_student_token_id": variant["student_token"],
                "chosen_token_id": first["token_id"],
                "chosen_token_text": first["token_text"],
                "chosen_logprob": first_logprob,
                "chosen_probability": math.exp(first_logprob)
                if first_logprob is not None
                else None,
                "top_logprobs_after_first_token_constraint": serialize_top_logprobs(
                    first["entry"], tokenizer, args.first_step_logprobs
                ),
            },
        }
        grouped.setdefault(index, []).append(artifact)
    return grouped


def student_process_batch(
    rows: list[dict[str, Any]],
    *,
    llm: Any,
    tokenizer: Any,
    args: argparse.Namespace,
) -> dict[int, list[dict[str, Any]]]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    variants: list[dict[str, Any]] = []
    prompts = []
    params = []
    for row in rows:
        bridge_rollouts = row["bridge"].get("rollouts")
        if not isinstance(bridge_rollouts, list) or len(bridge_rollouts) != args.num_rollouts:
            raise ValueError(
                f"trace {row['anchor']['dataset_index']}: invalid bridge rollout count"
            )
        index = int(row["anchor"]["dataset_index"])
        for rollout_index, bridge in enumerate(bridge_rollouts):
            full_bridge_ids = [int(token_id) for token_id in bridge["bridge_token_ids"]]
            if args.bridge_token_limit is not None:
                if len(full_bridge_ids) < args.bridge_token_limit:
                    raise ValueError(
                        f"trace {index}: bridge {rollout_index} has only "
                        f"{len(full_bridge_ids)} tokens, requested "
                        f"{args.bridge_token_limit}"
                    )
                bridge_ids = full_bridge_ids[: args.bridge_token_limit]
            else:
                bridge_ids = full_bridge_ids
            input_ids = [*row["prefix_ids"], *bridge_ids]
            max_tokens = args.max_model_len - len(input_ids)
            if max_tokens <= 0:
                raise ValueError(f"trace {index}: no room for student continuation")
            seed = args.seed + index * args.num_rollouts + rollout_index
            variant = {
                **row,
                "rollout_index": rollout_index,
                "bridge_ids": bridge_ids,
                "full_bridge_count": len(full_bridge_ids),
                "input_ids": input_ids,
                "seed": seed,
            }
            variants.append(variant)
            prompts.append(TokensPrompt(prompt_token_ids=input_ids))
            params.append(
                SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    seed=seed,
                    max_tokens=max_tokens,
                    n=1,
                    detokenize=False,
                    skip_special_tokens=False,
                )
            )

    outputs = llm.generate(prompts, params, use_tqdm=False)
    if len(outputs) != len(variants):
        raise RuntimeError("Student continuation output count mismatch")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for variant, request in zip(variants, outputs, strict=True):
        generated = request.outputs[0]
        continuation_ids = [int(token_id) for token_id in generated.token_ids]
        response_with_bridge = [
            *variant["response_prefix_ids"],
            *variant["bridge_ids"],
        ]
        full_trace_ids = [*response_with_bridge, *continuation_ids]
        continuation_text = tokenizer.decode(
            continuation_ids, skip_special_tokens=False
        )
        index = int(variant["anchor"]["dataset_index"])
        artifact = {
            "rollout_index": variant["rollout_index"],
            "student_seed": variant["seed"],
            "source_bridge_token_ids": variant["bridge_ids"],
            "source_bridge_text": tokenizer.decode(
                variant["bridge_ids"], skip_special_tokens=False
            ),
            "source_bridge_token_count": len(variant["bridge_ids"]),
            "source_bridge_original_token_count": variant["full_bridge_count"],
            "student_continuation_token_ids": continuation_ids,
            "student_continuation_text": continuation_text,
            "student_continuation_token_count": len(continuation_ids),
            "full_trace_token_ids": full_trace_ids,
            "full_trace_text": tokenizer.decode(
                full_trace_ids, skip_special_tokens=False
            ),
            "full_trace_token_count": len(full_trace_ids),
            "finish_reason": generated.finish_reason,
            "stop_reason": generated.stop_reason,
            "reached_model_length": generated.finish_reason == "length",
            "contains_thinking_tag": any(
                tag in continuation_text for tag in ("<think>", "</think>")
            ),
        }
        grouped.setdefault(index, []).append(artifact)
    return grouped


def student_prefix_process_batch(
    rows: list[dict[str, Any]],
    *,
    llm: Any,
    tokenizer: Any,
    args: argparse.Namespace,
) -> dict[int, list[dict[str, Any]]]:
    """Roll out the student directly after the prefix before the anchor token."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    variants: list[dict[str, Any]] = []
    prompts = []
    params = []
    for row in rows:
        index = int(row["anchor"]["dataset_index"])
        max_tokens = args.max_model_len - len(row["prefix_ids"])
        if max_tokens <= 0:
            raise ValueError(f"trace {index}: no room for student completion")
        for rollout_index in range(args.num_rollouts):
            seed = args.seed + index * args.num_rollouts + rollout_index
            variants.append(
                {
                    **row,
                    "rollout_index": rollout_index,
                    "seed": seed,
                }
            )
            prompts.append(TokensPrompt(prompt_token_ids=row["prefix_ids"]))
            params.append(
                SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    seed=seed,
                    max_tokens=max_tokens,
                    min_tokens=1,
                    n=1,
                    detokenize=False,
                    skip_special_tokens=False,
                )
            )

    outputs = llm.generate(prompts, params, use_tqdm=False)
    if len(outputs) != len(variants):
        raise RuntimeError("Student prefix completion output count mismatch")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for variant, request in zip(variants, outputs, strict=True):
        generated = request.outputs[0]
        completion_ids = [int(token_id) for token_id in generated.token_ids]
        if not completion_ids:
            raise RuntimeError(
                f"trace {variant['anchor']['dataset_index']}: empty student completion"
            )
        full_trace_ids = [*variant["response_prefix_ids"], *completion_ids]
        completion_text = tokenizer.decode(
            completion_ids, skip_special_tokens=False
        )
        index = int(variant["anchor"]["dataset_index"])
        artifact = {
            "rollout_index": variant["rollout_index"],
            "student_seed": variant["seed"],
            "student_completion_token_ids": completion_ids,
            "student_completion_text": completion_text,
            "student_completion_token_count": len(completion_ids),
            "first_token_matches_original_student_anchor": completion_ids[0]
            == int(variant["anchor"]["student_candidate_token_id"]),
            "full_trace_token_ids": full_trace_ids,
            "full_trace_text": tokenizer.decode(
                full_trace_ids, skip_special_tokens=False
            ),
            "full_trace_token_count": len(full_trace_ids),
            "finish_reason": generated.finish_reason,
            "stop_reason": generated.stop_reason,
            "reached_model_length": generated.finish_reason == "length",
            "contains_thinking_tag": any(
                tag in completion_text for tag in ("<think>", "</think>")
            ),
        }
        grouped.setdefault(index, []).append(artifact)
    return grouped


def make_bundle(
    row: dict[str, Any],
    rollouts: list[dict[str, Any]],
    args: argparse.Namespace,
    tokenizer: Any,
) -> dict[str, Any]:
    anchor = row["anchor"]
    rollouts.sort(key=lambda rollout: int(rollout["rollout_index"]))
    if len(rollouts) != args.num_rollouts:
        raise RuntimeError(
            f"trace {anchor['dataset_index']}: got {len(rollouts)} rollouts"
        )
    value = {
        "status": bundle_status(args.mode),
        "schema_version": 1,
        "mode": args.mode,
        "dataset_index": int(anchor["dataset_index"]),
        "problem_id": anchor.get("problem_id"),
        "trace_sha256": anchor.get("trace_sha256"),
        "anchor_artifact": str(row["anchor_path"]),
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "max_model_len": args.max_model_len,
        "requested_rollouts": args.num_rollouts,
        "actual_rollouts": len(rollouts),
        "prompt_token_count": row["prompt_token_count"],
        "prompt_text": tokenizer.decode(
            row["prefix_ids"][: row["prompt_token_count"]],
            skip_special_tokens=False,
        ),
        "response_prefix_before_anchor_token_ids": row["response_prefix_ids"],
        "response_prefix_before_anchor_text": tokenizer.decode(
            row["response_prefix_ids"], skip_special_tokens=False
        ),
        "student_anchor_token_id": int(anchor["student_candidate_token_id"]),
        "student_anchor_token_text": anchor["student_candidate_token_text"],
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "base_seed": args.seed,
            "stop_at_eos": True,
        },
        "rollouts": rollouts,
    }
    if args.mode == STUDENT_MODE:
        value["bridge_artifact"] = str(row["bridge_path"])
        value["source_bridge_token_limit"] = args.bridge_token_limit
    elif args.mode == TEACHER_MODE:
        value["all_first_tokens_differ_from_student"] = all(
            rollout["teacher_completion_token_ids"][0]
            != int(anchor["student_candidate_token_id"])
            for rollout in rollouts
        )
    else:
        value["first_token_matches_original_student_anchor_count"] = sum(
            bool(rollout["first_token_matches_original_student_anchor"])
            for rollout in rollouts
        )
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--anchor-root", type=Path)
    parser.add_argument("--bridge-manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--first-step-logprobs", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--trace-batch-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--bridge-token-limit",
        type=int,
        help="For student_after_bridge, condition on only the first N bridge tokens.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode in {TEACHER_MODE, STUDENT_PREFIX_MODE} and args.anchor_root is None:
        raise ValueError("--anchor-root is required for anchor-prefix modes")
    if args.mode == STUDENT_MODE and args.bridge_manifest is None:
        raise ValueError("--bridge-manifest is required for student mode")
    if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Invalid shard specification")
    if args.num_rollouts <= 0 or args.trace_batch_size <= 0:
        raise ValueError("Rollout count and trace batch size must be positive")
    if args.bridge_token_limit is not None:
        if args.mode != STUDENT_MODE:
            raise ValueError("--bridge-token-limit is only valid for student_after_bridge")
        if args.bridge_token_limit <= 0:
            raise ValueError("--bridge-token-limit must be positive")

    from transformers import AutoTokenizer
    from vllm import LLM

    started = time.time()
    all_rows = discover_records(args)
    records = shard_records(
        all_rows, args.shard_id, args.num_shards, args.max_records
    )
    if not records:
        raise ValueError(f"Shard {args.shard_id} has no records")
    args.output_root.mkdir(parents=True, exist_ok=True)
    shard_root = args.output_root / "shards" / f"shard_{args.shard_id:02d}"
    shard_root.mkdir(parents=True, exist_ok=True)

    pending = [
        row
        for row in records
        if not valid_cached_bundle(
            output_path(
                args.output_root,
                int(row["anchor"]["dataset_index"]),
                args.mode,
                args.num_rollouts,
            ),
            row,
            args.mode,
            args.model,
            args.num_rollouts,
            args.max_model_len,
            args.bridge_token_limit,
        )
    ]
    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    prepared = [prepare_common(row, args.max_model_len) for row in pending]
    prepared.sort(key=lambda row: len(row["prefix_ids"]))
    prefix_lengths = [len(row["prefix_ids"]) for row in prepared]
    print(
        "FULL_ROLLOUT_SHARD_PREFLIGHT "
        f"mode={args.mode} shard={args.shard_id}/{args.num_shards} "
        f"expected={len(records)} cached={len(records) - len(pending)} "
        f"pending={len(pending)} prefix_max={max(prefix_lengths) if prefix_lengths else 0}",
        flush=True,
    )

    generated = 0
    generated_tokens = 0
    if prepared:
        max_num_seqs = args.trace_batch_size * args.num_rollouts
        llm = LLM(
            model=args.model,
            tokenizer=tokenizer_path,
            tensor_parallel_size=1,
            dtype="bfloat16",
            trust_remote_code=False,
            seed=args.seed,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=max_num_seqs,
            enable_prefix_caching=True,
            disable_log_stats=True,
            max_logprobs=max(1, args.first_step_logprobs),
        )
        for offset in range(0, len(prepared), args.trace_batch_size):
            trace_batch = prepared[offset : offset + args.trace_batch_size]
            if args.mode == TEACHER_MODE:
                grouped = teacher_process_batch(
                    trace_batch, llm=llm, tokenizer=tokenizer, args=args
                )
            elif args.mode == STUDENT_MODE:
                grouped = student_process_batch(
                    trace_batch, llm=llm, tokenizer=tokenizer, args=args
                )
            else:
                grouped = student_prefix_process_batch(
                    trace_batch, llm=llm, tokenizer=tokenizer, args=args
                )
            for row in trace_batch:
                index = int(row["anchor"]["dataset_index"])
                rollouts = grouped.get(index, [])
                bundle = make_bundle(row, rollouts, args, tokenizer)
                atomic_write_json(
                    output_path(
                        args.output_root, index, args.mode, args.num_rollouts
                    ),
                    bundle,
                )
                generated += 1
                if args.mode == TEACHER_MODE:
                    generated_tokens += sum(
                        rollout["teacher_completion_token_count"]
                        for rollout in rollouts
                    )
                elif args.mode == STUDENT_MODE:
                    generated_tokens += sum(
                        rollout["student_continuation_token_count"]
                        for rollout in rollouts
                    )
                else:
                    generated_tokens += sum(
                        rollout["student_completion_token_count"]
                        for rollout in rollouts
                    )
            print(
                "FULL_ROLLOUT_SHARD_PROGRESS "
                f"mode={args.mode} shard={args.shard_id} "
                f"completed={len(records) - len(pending) + generated}/{len(records)} "
                f"generated_now={generated} generated_tokens={generated_tokens}",
                flush=True,
            )

    summary = {
        "status": "FULL_ROLLOUT_SHARD_COMPLETE",
        "mode": args.mode,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "expected_records": len(records),
        "cached_records": len(records) - len(pending),
        "generated_records": generated,
        "generated_rollouts": generated * args.num_rollouts,
        "generated_tokens": generated_tokens,
        "requested_rollouts_per_trace": args.num_rollouts,
        "model": args.model,
        "trace_batch_size": args.trace_batch_size,
        "max_model_len": args.max_model_len,
        "source_bridge_token_limit": args.bridge_token_limit
        if args.mode == STUDENT_MODE
        else None,
        "prefix_length_max": max(prefix_lengths) if prefix_lengths else None,
        "prefix_length_median": statistics.median(prefix_lengths)
        if prefix_lengths
        else None,
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    atomic_write_json(shard_root / "summary.json", summary)
    (shard_root / "SHARD_COMPLETE").touch()
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
