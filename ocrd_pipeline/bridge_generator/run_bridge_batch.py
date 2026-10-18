#!/usr/bin/env python3
"""Generate exact-length teacher bridges for one deterministic anchor shard."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR.parent))

from bridge_generator.core import (  # noqa: E402
    assemble_bridge_artifact,
    atomic_write_json,
    load_selected_anchor,
)
from bridge_generator.run_bridge_generator import (  # noqa: E402
    chosen_logprob,
    serialize_top_logprobs,
)


DEFAULT_TEACHER = "/storage/yukaichengLab/share/model/Qwen3-14B"


def discover_selected_anchors(anchor_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    candidates = list(anchor_root.glob("trace_*/selected_anchor.json"))
    candidates.extend(anchor_root.glob("traces/trace_*/selected_anchor.json"))
    for path in candidates:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") == "ANCHOR_SELECTED":
            rows.append((path, load_selected_anchor(path)))
    rows.sort(key=lambda item: int(item[1]["dataset_index"]))
    return rows


def shard_records(
    rows: list[tuple[Path, dict[str, Any]]],
    shard_id: int,
    num_shards: int,
    max_records: int | None,
) -> list[tuple[Path, dict[str, Any]]]:
    selected = [
        item
        for item in rows
        if int(item[1]["dataset_index"]) % num_shards == shard_id
    ]
    if max_records is not None:
        selected = selected[:max_records]
    return selected


def output_path(
    output_root: Path, dataset_index: int, h: int, num_rollouts: int
) -> Path:
    return (
        output_root
        / "traces"
        / f"trace_{dataset_index:05d}"
        / f"bridge_h{h}_n{num_rollouts}.json"
    )


def valid_cached_bundle(
    path: Path,
    anchor: dict[str, Any],
    requested_length: int,
    num_rollouts: int,
    teacher: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    rollouts = value.get("rollouts")
    if not bool(
        value.get("status") == "BRIDGE_ROLLOUTS_GENERATED"
        and value.get("trace_sha256") == anchor.get("trace_sha256")
        and value.get("teacher_model") == teacher
        and value.get("requested_bridge_length") == requested_length
        and value.get("requested_rollouts") == num_rollouts
        and value.get("actual_rollouts") == num_rollouts
        and isinstance(rollouts, list)
        and len(rollouts) == num_rollouts
    ):
        return False
    student = int(anchor["student_candidate_token_id"])
    for rollout_index, rollout in enumerate(rollouts):
        bridge_ids = rollout.get("bridge_token_ids")
        if not bool(
            rollout.get("status") == "BRIDGE_GENERATED"
            and rollout.get("rollout_index") == rollout_index
            and rollout.get("actual_bridge_length") == requested_length
            and isinstance(bridge_ids, list)
            and len(bridge_ids) == requested_length
            and bridge_ids
            and int(bridge_ids[0]) != student
            and rollout.get("first_token_differs_from_student") is True
        ):
            return False
    return True


def prepare_record(
    path: Path,
    anchor: dict[str, Any],
    tokenizer: Any,
    h: int,
    max_model_len: int,
) -> dict[str, Any]:
    prefix_ids = [int(token_id) for token_id in anchor["anchor_prefix_token_ids"]]
    if len(prefix_ids) + h > max_model_len:
        raise ValueError(
            f"trace {anchor['dataset_index']}: prefix={len(prefix_ids)} + h={h} "
            f"exceeds max_model_len={max_model_len}"
        )
    student_token_id = int(anchor["student_candidate_token_id"])
    if not 0 <= student_token_id < len(tokenizer):
        raise ValueError(
            f"trace {anchor['dataset_index']}: student token {student_token_id} "
            f"outside tokenizer vocabulary of size {len(tokenizer)}"
        )
    student_token_text = tokenizer.decode(
        [student_token_id], skip_special_tokens=False
    )
    # Use vLLM's already-tokenized bad-word field. A decoded token does not
    # necessarily round-trip to the same single token, so passing bad_words
    # as text would fail to exclude some student token IDs.
    bad_sequences = [[student_token_id]]
    return {
        "path": path,
        "anchor": anchor,
        "prefix_ids": prefix_ids,
        "student_token_id": student_token_id,
        "student_token_text": student_token_text,
        "bad_sequences": bad_sequences,
    }


def process_batch(
    batch: list[dict[str, Any]],
    *,
    llm: Any,
    tokenizer: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    prompts = [TokensPrompt(prompt_token_ids=row["prefix_ids"]) for row in batch]
    first_params = [
        SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=row["first_seed"],
            max_tokens=1,
            min_tokens=1,
            ignore_eos=True,
            logprobs=args.first_step_logprobs,
            _bad_words_token_ids=row["bad_sequences"],
            detokenize=False,
            skip_special_tokens=False,
        )
        for row in batch
    ]
    first_requests = llm.generate(prompts, first_params, use_tqdm=False)
    if len(first_requests) != len(batch):
        raise RuntimeError("First-step batch output count mismatch")

    first_rows: list[dict[str, Any]] = []
    continuation_prompts = []
    continuation_params = []
    for row, request in zip(batch, first_requests, strict=True):
        output = request.outputs[0]
        token_ids = [int(token_id) for token_id in output.token_ids]
        if len(token_ids) != 1:
            raise RuntimeError(
                f"trace {row['anchor']['dataset_index']}: first step produced "
                f"{len(token_ids)} tokens"
            )
        first_id = token_ids[0]
        if first_id == row["student_token_id"]:
            raise RuntimeError(
                f"trace {row['anchor']['dataset_index']}: first-token constraint violated"
            )
        entry = output.logprobs[0] if output.logprobs else None
        first_rows.append(
            {
                "id": first_id,
                "output": output,
                "entry": entry,
                "logprob": chosen_logprob(entry, first_id),
            }
        )
        continuation_prompts.append(
            TokensPrompt(prompt_token_ids=row["prefix_ids"] + [first_id])
        )
        continuation_params.append(
            SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=row["continuation_seed"],
                max_tokens=args.h - 1,
                min_tokens=args.h - 1,
                ignore_eos=True,
                logprobs=1,
                detokenize=False,
                skip_special_tokens=False,
            )
        )

    continuation_requests = llm.generate(
        continuation_prompts, continuation_params, use_tqdm=False
    )
    if len(continuation_requests) != len(batch):
        raise RuntimeError("Continuation batch output count mismatch")

    artifacts: list[dict[str, Any]] = []
    for row, first, request in zip(
        batch, first_rows, continuation_requests, strict=True
    ):
        continuation = request.outputs[0]
        continuation_ids = [int(token_id) for token_id in continuation.token_ids]
        if len(continuation_ids) != args.h - 1:
            raise RuntimeError(
                f"trace {row['anchor']['dataset_index']}: continuation produced "
                f"{len(continuation_ids)} tokens, expected {args.h - 1}"
            )
        bridge_ids = [first["id"], *continuation_ids]
        entries = continuation.logprobs or [None] * (args.h - 1)
        chosen = [first["logprob"]]
        chosen.extend(
            chosen_logprob(entry, token_id)
            for entry, token_id in zip(entries, continuation_ids, strict=True)
        )
        token_texts = [
            tokenizer.decode([token_id], skip_special_tokens=False)
            for token_id in bridge_ids
        ]
        dataset_index = int(row["anchor"]["dataset_index"])
        artifact = assemble_bridge_artifact(
                anchor=row["anchor"],
                anchor_path=row["path"],
                teacher_model=args.teacher,
                tokenizer_path=args.tokenizer or args.teacher,
                requested_length=args.h,
                bridge_token_ids=bridge_ids,
                bridge_token_texts=token_texts,
                bridge_text=tokenizer.decode(
                    bridge_ids, skip_special_tokens=False
                ),
                chosen_logprobs=chosen,
                sampling={
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "first_step_seed": row["first_seed"],
                    "continuation_seed": row["continuation_seed"],
                    "ignore_eos": True,
                    "exact_length_enforced_by_min_tokens_equals_max_tokens": True,
                },
                first_step={
                    "finish_reason": first["output"].finish_reason,
                    "chosen_token_id": first["id"],
                    "chosen_token_text": token_texts[0],
                    "chosen_logprob": first["logprob"],
                    "top_logprobs_after_first_token_constraint": serialize_top_logprobs(
                        first["entry"], tokenizer, args.first_step_logprobs
                    ),
                    "bad_word_token_sequences": row["bad_sequences"],
                },
                continuation_finish_reason=continuation.finish_reason,
            )
        artifact["rollout_index"] = row["rollout_index"]
        artifacts.append(artifact)
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--h", type=int, default=256)
    parser.add_argument("--teacher", default=DEFAULT_TEACHER)
    parser.add_argument("--tokenizer")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--first-step-logprobs", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--trace-batch-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.h < 2:
        raise ValueError("Batch bridge generation requires --h >= 2")
    if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Invalid shard specification")
    if args.num_rollouts <= 0:
        raise ValueError("--num-rollouts must be positive")
    if args.trace_batch_size <= 0:
        raise ValueError("--trace-batch-size must be positive")

    from transformers import AutoTokenizer
    from vllm import LLM

    started = time.time()
    all_rows = discover_selected_anchors(args.anchor_root)
    records = shard_records(
        all_rows, args.shard_id, args.num_shards, args.max_records
    )
    if not records:
        raise ValueError(f"Shard {args.shard_id} has no selected anchors")
    args.output_root.mkdir(parents=True, exist_ok=True)
    shard_root = args.output_root / "shards" / f"shard_{args.shard_id:02d}"
    shard_root.mkdir(parents=True, exist_ok=True)

    pending = [
        (path, anchor)
        for path, anchor in records
        if not valid_cached_bundle(
            output_path(
                args.output_root,
                int(anchor["dataset_index"]),
                args.h,
                args.num_rollouts,
            ),
            anchor,
            args.h,
            args.num_rollouts,
            args.teacher,
        )
    ]
    tokenizer_path = args.tokenizer or args.teacher
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    prepared = [
        prepare_record(path, anchor, tokenizer, args.h, args.max_model_len)
        for path, anchor in pending
    ]
    prepared.sort(key=lambda row: len(row["prefix_ids"]))
    prefix_lengths = [len(row["prefix_ids"]) for row in prepared]
    print(
        "BRIDGE_SHARD_PREFLIGHT "
        f"shard={args.shard_id}/{args.num_shards} expected={len(records)} "
        f"cached={len(records) - len(pending)} pending={len(pending)} "
        f"prefix_max={max(prefix_lengths) if prefix_lengths else 0}",
        flush=True,
    )

    generated = 0
    if prepared:
        llm = LLM(
            model=args.teacher,
            tokenizer=tokenizer_path,
            tensor_parallel_size=1,
            dtype="bfloat16",
            trust_remote_code=False,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.trace_batch_size * args.num_rollouts,
            enable_prefix_caching=True,
            disable_log_stats=True,
            max_logprobs=max(1, args.first_step_logprobs),
        )
        for offset in range(0, len(prepared), args.trace_batch_size):
            trace_batch = prepared[offset : offset + args.trace_batch_size]
            batch: list[dict[str, Any]] = []
            for row in trace_batch:
                dataset_index = int(row["anchor"]["dataset_index"])
                for rollout_index in range(args.num_rollouts):
                    variant = dict(row)
                    variant["rollout_index"] = rollout_index
                    variant["first_seed"] = args.seed + 2 * (
                        dataset_index * args.num_rollouts + rollout_index
                    )
                    variant["continuation_seed"] = variant["first_seed"] + 1
                    batch.append(variant)
            artifacts = process_batch(
                batch, llm=llm, tokenizer=tokenizer, args=args
            )
            for row in trace_batch:
                index = int(row["anchor"]["dataset_index"])
                trace_rollouts = [
                    artifact
                    for artifact in artifacts
                    if int(artifact["dataset_index"]) == index
                ]
                trace_rollouts.sort(key=lambda artifact: artifact["rollout_index"])
                if len(trace_rollouts) != args.num_rollouts:
                    raise RuntimeError(
                        f"trace {index}: got {len(trace_rollouts)} rollouts, "
                        f"expected {args.num_rollouts}"
                    )
                bundle = {
                    "status": "BRIDGE_ROLLOUTS_GENERATED",
                    "schema_version": 1,
                    "dataset_index": index,
                    "problem_id": row["anchor"].get("problem_id"),
                    "trace_sha256": row["anchor"].get("trace_sha256"),
                    "anchor_artifact": str(row["path"]),
                    "teacher_model": args.teacher,
                    "tokenizer": tokenizer_path,
                    "requested_bridge_length": args.h,
                    "requested_rollouts": args.num_rollouts,
                    "actual_rollouts": len(trace_rollouts),
                    "all_first_tokens_differ_from_student": all(
                        artifact["first_token_differs_from_student"]
                        for artifact in trace_rollouts
                    ),
                    "rollouts": trace_rollouts,
                }
                atomic_write_json(
                    output_path(
                        args.output_root, index, args.h, args.num_rollouts
                    ),
                    bundle,
                )
                generated += 1
            print(
                "BRIDGE_SHARD_PROGRESS "
                f"shard={args.shard_id} completed={len(records) - len(pending) + generated}/"
                f"{len(records)} generated_now={generated}",
                flush=True,
            )

    summary = {
        "status": "BRIDGE_SHARD_COMPLETE",
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "expected_records": len(records),
        "cached_records": len(records) - len(pending),
        "generated_records": generated,
        "requested_rollouts_per_trace": args.num_rollouts,
        "generated_bridges": generated * args.num_rollouts,
        "requested_bridge_length": args.h,
        "teacher_model": args.teacher,
        "trace_batch_size": args.trace_batch_size,
        "max_num_seqs": args.trace_batch_size * args.num_rollouts,
        "prefix_length_max": max(prefix_lengths) if prefix_lengths else None,
        "prefix_length_median": statistics.median(prefix_lengths)
        if prefix_lengths
        else None,
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(shard_root / "summary.json", summary)
    (shard_root / "SHARD_COMPLETE").touch()
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
