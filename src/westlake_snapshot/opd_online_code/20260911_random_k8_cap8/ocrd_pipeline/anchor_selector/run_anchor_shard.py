#!/usr/bin/env python3
"""GPU-efficient anchor candidate extraction for one rollout shard.

The two phases are intentionally separate processes.  The teacher phase loads
Qwen3-14B once and writes resumable top-k caches.  The student phase loads
Qwen3-1.7B once, computes the chosen-token log probabilities, and materializes
self-contained candidate records for the network judge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR.parent))

from anchor_selector.core import (  # noqa: E402
    build_candidates,
    lookup_actual_logprob,
    sorted_topk,
    tokenizer_fingerprint,
    trace_identity,
)


DEFAULT_STUDENT = "/storage/yukaichengLab/share/model/Qwen3-1.7B"
DEFAULT_TEACHER = "/storage/yukaichengLab/share/model/Qwen3-14B"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any, *, pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
        )
        handle.write("\n")
    temporary.replace(path)


def atomic_npz(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


def read_rows(path: Path, include_padding: bool, limit: int | None) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not include_padding:
        rows = [row for row in rows if not bool(row.get("padded_duplicate", False))]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No eligible rows in {path}")
    seen: set[int] = set()
    for row in rows:
        for field in ("prompt_token_ids", "response_token_ids"):
            values = row.get(field)
            if not isinstance(values, list) or not values:
                raise ValueError(f"dataset_index={row.get('dataset_index')} has invalid {field}")
            if not all(isinstance(value, int) and value >= 0 for value in values):
                raise TypeError(f"dataset_index={row.get('dataset_index')} has invalid {field}")
        index = int(row["dataset_index"])
        if index in seen:
            raise ValueError(f"Duplicate dataset_index={index} within shard")
        seen.add(index)
    return rows


def cache_path(output_dir: Path, dataset_index: int) -> Path:
    return output_dir / "teacher_scores" / f"trace_{dataset_index:05d}.npz"


def record_path(output_dir: Path, dataset_index: int) -> Path:
    return output_dir / "candidate_records" / f"trace_{dataset_index:05d}.json"


def valid_teacher_cache(path: Path, response_tokens: int, topk: int) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path) as values:
            return (
                values["chosen_logprobs"].shape == (response_tokens,)
                and values["topk_ids"].shape == (response_tokens, topk)
                and values["topk_logprobs"].shape == (response_tokens, topk)
                and np.isfinite(values["chosen_logprobs"]).all()
                and np.isfinite(values["topk_logprobs"]).all()
            )
    except (OSError, KeyError, ValueError):
        return False


def valid_candidate_record(path: Path, trace: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return (
            value.get("status") == "ANCHOR_CANDIDATES_OK"
            and value.get("trace_sha256") == trace_identity(trace)["trace_sha256"]
            and int(value.get("candidate_count", -1)) == len(value.get("candidates", []))
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def command_teacher(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    rows = read_rows(args.trace_shard, args.include_padding, args.limit)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    topk = int(args.topk or config["teacher_topk"])
    tokenizer_path = args.tokenizer or args.teacher
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    fingerprint = tokenizer_fingerprint(tokenizer)
    for row in rows:
        sequence = row["prompt_token_ids"] + row["response_token_ids"]
        if len(sequence) + 1 > args.max_model_len:
            raise ValueError(
                f"dataset_index={row['dataset_index']} has {len(sequence)} input tokens; "
                f"max_model_len={args.max_model_len} must also fit one generated token"
            )
        if max(sequence) >= len(tokenizer):
            raise ValueError(f"dataset_index={row['dataset_index']} has token outside vocabulary")

    engine = LLM(
        model=args.teacher,
        tokenizer=tokenizer_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=False,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
        enable_prefix_caching=False,
        disable_log_stats=True,
        max_logprobs=topk,
    )
    sampling = SamplingParams(
        temperature=1.0,
        max_tokens=1,
        prompt_logprobs=topk,
        seed=args.seed,
    )
    started = time.time()
    skipped = 0
    total_tokens = 0
    pending: list[dict[str, Any]] = []
    for row in rows:
        response_length = len(row["response_token_ids"])
        output = cache_path(args.output_dir, int(row["dataset_index"]))
        if args.resume and valid_teacher_cache(output, response_length, topk):
            skipped += 1
            total_tokens += response_length
            continue
        pending.append(row)

    completed = skipped
    for batch_start in range(0, len(pending), args.batch_size):
        batch_rows = pending[batch_start : batch_start + args.batch_size]
        sequences = [
            row["prompt_token_ids"] + row["response_token_ids"] for row in batch_rows
        ]
        requests = engine.generate(
            [TokensPrompt(prompt_token_ids=sequence) for sequence in sequences],
            sampling,
            use_tqdm=False,
        )
        for row, sequence, request in zip(batch_rows, sequences, requests, strict=True):
            response_length = len(row["response_token_ids"])
            output = cache_path(args.output_dir, int(row["dataset_index"]))
            prompt_logprobs = request.prompt_logprobs
            if prompt_logprobs is None or len(prompt_logprobs) != len(sequence):
                raise RuntimeError(
                    f"dataset_index={row['dataset_index']} received misaligned prompt logprobs"
                )
            response_start = len(row["prompt_token_ids"])
            chosen = np.empty(response_length, dtype=np.float32)
            top_ids = np.empty((response_length, topk), dtype=np.int32)
            top_logprobs = np.empty((response_length, topk), dtype=np.float32)
            for position, token_id in enumerate(row["response_token_ids"]):
                entry = prompt_logprobs[response_start + position]
                chosen[position] = lookup_actual_logprob(entry, int(token_id))
                ids, logprobs = sorted_topk(entry, topk, len(tokenizer))
                top_ids[position] = ids
                top_logprobs[position] = logprobs
            atomic_npz(
                output,
                chosen_logprobs=chosen,
                topk_ids=top_ids,
                topk_logprobs=top_logprobs,
            )
            total_tokens += response_length
            completed += 1
            del prompt_logprobs, chosen, top_ids, top_logprobs
        if completed % args.progress_every < len(batch_rows) or completed == len(rows):
            print(
                f"TEACHER_PROGRESS rows={completed}/{len(rows)} response_tokens={total_tokens} "
                f"elapsed_seconds={time.time() - started:.1f}",
                flush=True,
            )
        del requests, sequences, batch_rows

    manifest = {
        "status": "SHARD_TEACHER_COMPLETE",
        "backend": "vllm_prompt_logprobs",
        "trace_shard": str(args.trace_shard),
        "trace_shard_sha256": sha256_file(args.trace_shard),
        "rows": len(rows),
        "response_tokens": sum(len(row["response_token_ids"]) for row in rows),
        "padding_included": bool(args.include_padding),
        "teacher": args.teacher,
        "tokenizer": tokenizer_path,
        "vocab_size": len(tokenizer),
        "tokenizer_vocab_sha256": fingerprint,
        "topk": topk,
        "max_model_len": args.max_model_len,
        "elapsed_seconds": time.time() - started,
        "resumed_rows": skipped,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "node": os.environ.get("SLURMD_NODENAME"),
    }
    atomic_json(args.output_dir / "teacher_manifest.json", manifest)
    (args.output_dir / "SHARD_TEACHER_COMPLETE").touch()
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)


def command_teacher_transformers(args: argparse.Namespace) -> None:
    """Score a shard without vLLM's per-token Python logprob dictionaries."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.logit_chunk_size <= 0:
        raise ValueError("--logit-chunk-size must be positive")
    rows = read_rows(args.trace_shard, args.include_padding, args.limit)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    topk = int(args.topk or config["teacher_topk"])
    tokenizer_path = args.tokenizer or args.teacher
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    fingerprint = tokenizer_fingerprint(tokenizer)
    for row in rows:
        sequence = row["prompt_token_ids"] + row["response_token_ids"]
        if len(sequence) > args.max_model_len:
            raise ValueError(
                f"dataset_index={row['dataset_index']} has {len(sequence)} input tokens; "
                f"max_model_len={args.max_model_len}"
            )
        if max(sequence) >= len(tokenizer):
            raise ValueError(f"dataset_index={row['dataset_index']} has token outside vocabulary")

    model = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation=args.attention_implementation,
    ).to("cuda")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    started = time.time()
    skipped = 0
    total_tokens = 0
    with torch.inference_mode():
        for ordinal, row in enumerate(rows, start=1):
            response_ids = row["response_token_ids"]
            response_length = len(response_ids)
            output = cache_path(args.output_dir, int(row["dataset_index"]))
            if args.resume and valid_teacher_cache(output, response_length, topk):
                skipped += 1
                total_tokens += response_length
                continue
            sequence = row["prompt_token_ids"] + response_ids
            input_ids = torch.tensor(sequence, dtype=torch.long, device="cuda").unsqueeze(0)
            base_output = model.model(input_ids=input_ids, use_cache=False, return_dict=True)
            hidden = base_output.last_hidden_state[0]
            prediction_start = len(row["prompt_token_ids"]) - 1
            chosen = np.empty(response_length, dtype=np.float32)
            top_ids = np.empty((response_length, topk), dtype=np.int32)
            top_logprobs = np.empty((response_length, topk), dtype=np.float32)
            for chunk_start in range(0, response_length, args.logit_chunk_size):
                chunk_stop = min(response_length, chunk_start + args.logit_chunk_size)
                positions = torch.arange(
                    prediction_start + chunk_start,
                    prediction_start + chunk_stop,
                    dtype=torch.long,
                    device="cuda",
                )
                logits = model.lm_head(hidden.index_select(0, positions)).float()
                log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
                chosen_ids = torch.tensor(
                    response_ids[chunk_start:chunk_stop],
                    dtype=torch.long,
                    device="cuda",
                )
                chosen_values = (
                    logits.gather(1, chosen_ids[:, None]).squeeze(1) - log_z.squeeze(1)
                )
                top_values, top_indices = torch.topk(logits - log_z, k=topk, dim=-1)
                chosen[chunk_start:chunk_stop] = chosen_values.cpu().numpy()
                top_ids[chunk_start:chunk_stop] = top_indices.cpu().numpy()
                top_logprobs[chunk_start:chunk_stop] = top_values.cpu().numpy()
                del (
                    positions,
                    logits,
                    log_z,
                    chosen_ids,
                    chosen_values,
                    top_values,
                    top_indices,
                )
            atomic_npz(
                output,
                chosen_logprobs=chosen,
                topk_ids=top_ids,
                topk_logprobs=top_logprobs,
            )
            total_tokens += response_length
            if ordinal % args.progress_every == 0 or ordinal == len(rows):
                print(
                    f"TEACHER_TF_PROGRESS rows={ordinal}/{len(rows)} "
                    f"response_tokens={total_tokens} elapsed_seconds={time.time() - started:.1f}",
                    flush=True,
                )
            del input_ids, base_output, hidden, chosen, top_ids, top_logprobs

    manifest = {
        "status": "SHARD_TEACHER_COMPLETE",
        "backend": "transformers_hidden_lm_head",
        "trace_shard": str(args.trace_shard),
        "trace_shard_sha256": sha256_file(args.trace_shard),
        "rows": len(rows),
        "response_tokens": sum(len(row["response_token_ids"]) for row in rows),
        "padding_included": bool(args.include_padding),
        "teacher": args.teacher,
        "tokenizer": tokenizer_path,
        "vocab_size": len(tokenizer),
        "tokenizer_vocab_sha256": fingerprint,
        "topk": topk,
        "max_model_len": args.max_model_len,
        "logit_chunk_size": args.logit_chunk_size,
        "attention_implementation": args.attention_implementation,
        "elapsed_seconds": time.time() - started,
        "resumed_rows": skipped,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "node": os.environ.get("SLURMD_NODENAME"),
    }
    atomic_json(args.output_dir / "teacher_manifest.json", manifest)
    (args.output_dir / "SHARD_TEACHER_COMPLETE").touch()
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)


def student_chosen_logprobs(
    model: Any,
    hidden: Any,
    response_ids: list[int],
    prediction_start: int,
    chunk_size: int,
) -> np.ndarray:
    import torch

    result: list[np.ndarray] = []
    for chunk_start in range(0, len(response_ids), chunk_size):
        chunk_stop = min(len(response_ids), chunk_start + chunk_size)
        positions = torch.arange(
            prediction_start + chunk_start,
            prediction_start + chunk_stop,
            dtype=torch.long,
            device=hidden.device,
        )
        logits = model.lm_head(hidden.index_select(0, positions)).float()
        chosen_ids = torch.tensor(
            response_ids[chunk_start:chunk_stop], dtype=torch.long, device=hidden.device
        )
        values = logits.gather(1, chosen_ids[:, None]).squeeze(1) - torch.logsumexp(
            logits, dim=-1
        )
        result.append(values.cpu().numpy().astype(np.float32, copy=False))
        del positions, logits, chosen_ids, values
    return np.concatenate(result)


def command_student_candidates(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = read_rows(args.trace_shard, args.include_padding, args.limit)
    teacher_manifest_path = args.output_dir / "teacher_manifest.json"
    if not teacher_manifest_path.is_file():
        raise FileNotFoundError(f"Missing teacher manifest: {teacher_manifest_path}")
    teacher_manifest = json.loads(teacher_manifest_path.read_text(encoding="utf-8"))
    if teacher_manifest.get("status") != "SHARD_TEACHER_COMPLETE":
        raise ValueError("Teacher shard is not complete")
    if teacher_manifest.get("trace_shard_sha256") != sha256_file(args.trace_shard):
        raise ValueError("Teacher manifest belongs to a different trace shard")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    threshold = float(args.gap_threshold or config["gap_threshold"])
    topk = int(teacher_manifest["topk"])
    tokenizer_path = args.tokenizer or args.student
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    fingerprint = tokenizer_fingerprint(tokenizer)
    if len(tokenizer) != int(teacher_manifest["vocab_size"]):
        raise ValueError("Student and teacher tokenizer vocabulary sizes differ")
    if fingerprint != teacher_manifest["tokenizer_vocab_sha256"]:
        raise ValueError("Student and teacher token-to-ID maps differ")

    model = AutoModelForCausalLM.from_pretrained(
        args.student,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation=args.attention_implementation,
    ).to("cuda")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    started = time.time()
    skipped = 0
    total_tokens = 0
    total_candidates = 0
    with torch.inference_mode():
        for ordinal, row in enumerate(rows, start=1):
            index = int(row["dataset_index"])
            output = record_path(args.output_dir, index)
            if args.resume and valid_candidate_record(output, row):
                previous = json.loads(output.read_text(encoding="utf-8"))
                total_candidates += int(previous["candidate_count"])
                total_tokens += len(row["response_token_ids"])
                skipped += 1
                continue
            teacher_path = cache_path(args.output_dir, index)
            if not valid_teacher_cache(teacher_path, len(row["response_token_ids"]), topk):
                raise ValueError(f"Invalid teacher cache: {teacher_path}")
            with np.load(teacher_path) as values:
                teacher_chosen = values["chosen_logprobs"].copy()
                teacher_ids = values["topk_ids"].copy()
                teacher_logprobs = values["topk_logprobs"].copy()
            sequence = row["prompt_token_ids"] + row["response_token_ids"]
            input_ids = torch.tensor(sequence, dtype=torch.long, device="cuda").unsqueeze(0)
            base_output = model.model(input_ids=input_ids, use_cache=False, return_dict=True)
            hidden = base_output.last_hidden_state[0]
            student_chosen = student_chosen_logprobs(
                model,
                hidden,
                row["response_token_ids"],
                len(row["prompt_token_ids"]) - 1,
                args.logit_chunk_size,
            )
            candidates = build_candidates(
                row,
                tokenizer,
                student_chosen,
                teacher_chosen,
                teacher_ids,
                teacher_logprobs,
                threshold=threshold,
            )
            identity = trace_identity(row)
            record = {
                "status": "ANCHOR_CANDIDATES_OK",
                **identity,
                "padded_duplicate": bool(row.get("padded_duplicate", False)),
                "difficulty": row.get("difficulty"),
                "topic": row.get("topic"),
                "formula": "abs(log_pi_student_y_t - log_pi_teacher_y_t)",
                "strict_comparison": "absolute_gap > threshold",
                "threshold": threshold,
                "candidate_count": len(candidates),
                "trace": row,
                "candidates": candidates,
            }
            atomic_json(output, record, pretty=False)
            total_candidates += len(candidates)
            total_tokens += len(row["response_token_ids"])
            if ordinal % args.progress_every == 0 or ordinal == len(rows):
                print(
                    f"STUDENT_PROGRESS rows={ordinal}/{len(rows)} response_tokens={total_tokens} "
                    f"candidates={total_candidates} elapsed_seconds={time.time() - started:.1f}",
                    flush=True,
                )
            del (
                input_ids,
                base_output,
                hidden,
                student_chosen,
                teacher_chosen,
                teacher_ids,
                teacher_logprobs,
                candidates,
                record,
            )

    manifest = {
        "status": "SHARD_CANDIDATES_COMPLETE",
        "trace_shard": str(args.trace_shard),
        "trace_shard_sha256": sha256_file(args.trace_shard),
        "rows": len(rows),
        "response_tokens": sum(len(row["response_token_ids"]) for row in rows),
        "padding_included": bool(args.include_padding),
        "student": args.student,
        "tokenizer": tokenizer_path,
        "vocab_size": len(tokenizer),
        "tokenizer_vocab_sha256": fingerprint,
        "threshold": threshold,
        "candidate_count": total_candidates,
        "elapsed_seconds": time.time() - started,
        "resumed_rows": skipped,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "node": os.environ.get("SLURMD_NODENAME"),
    }
    atomic_json(args.output_dir / "candidate_manifest.json", manifest)
    (args.output_dir / "SHARD_CANDIDATES_COMPLETE").touch()
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trace-shard", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=MODULE_DIR / "config.json")
    parser.add_argument("--tokenizer")
    parser.add_argument("--include-padding", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--progress-every", type=int, default=25)


def parse_args() -> argparse.Namespace:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    teacher = commands.add_parser("teacher")
    add_common(teacher)
    teacher.add_argument("--teacher", default=DEFAULT_TEACHER)
    teacher.add_argument("--topk", type=int)
    teacher.add_argument("--max-model-len", type=int, default=40960)
    teacher.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    teacher.add_argument("--batch-size", type=int, default=8)
    teacher.set_defaults(func=command_teacher)

    teacher_tf = commands.add_parser("teacher-transformers")
    add_common(teacher_tf)
    teacher_tf.add_argument("--teacher", default=DEFAULT_TEACHER)
    teacher_tf.add_argument("--topk", type=int)
    teacher_tf.add_argument("--max-model-len", type=int, default=40960)
    teacher_tf.add_argument("--logit-chunk-size", type=int, default=256)
    teacher_tf.add_argument(
        "--attention-implementation",
        choices=["sdpa", "flash_attention_2"],
        default="sdpa",
    )
    teacher_tf.set_defaults(func=command_teacher_transformers)

    student = commands.add_parser("student-candidates")
    add_common(student)
    student.add_argument("--student", default=DEFAULT_STUDENT)
    student.add_argument("--gap-threshold", type=float)
    student.add_argument("--logit-chunk-size", type=int, default=256)
    student.add_argument(
        "--attention-implementation", choices=["sdpa", "flash_attention_2"], default="sdpa"
    )
    student.set_defaults(func=command_student_candidates)
    return root.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
