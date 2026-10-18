#!/usr/bin/env python3
"""Teacher-forced Student-Top16 reverse-KL training on all saved bridges.

The loss support at every bridge-token position is the current Student's
top-16 vocabulary items. Student and Teacher probabilities are renormalized
on that support and KL(Student || Teacher) is averaged over bridge tokens.
Question/prefix positions provide context only and are excluded from the loss.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import run_deepmath_full_vocab_reverse_kl_train as base  # noqa: E402


BRIDGE_ROOT = Path(os.environ.get(
    "OPD_BRIDGE_ROOT",
    "/storage/yukaichengLab/wangxin/test_eiloeo/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/bridge_qwen3_14b_h256_n8_seed20260822",
))
BRIDGE_MANIFEST = BRIDGE_ROOT / "manifest.jsonl"
TRACE_COUNT = int(os.environ.get("OPD_BRIDGE_TRACE_COUNT", "13244"))
BRIDGES_PER_TRACE = int(os.environ.get("OPD_BRIDGES_PER_TRACE", "8"))
BRIDGE_TOKENS = int(os.environ.get("OPD_BRIDGE_TOKENS", "256"))
SOURCE_BRIDGE_ROWS = TRACE_COUNT * BRIDGES_PER_TRACE
SELECTION_PATH = Path(os.environ["OPD_BRIDGE_SELECTION"]) if os.environ.get("OPD_BRIDGE_SELECTION") else None
if SELECTION_PATH is not None:
    _selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    if _selection.get("status") != "GOOD_BRIDGE_SELECTION_COMPLETE":
        raise RuntimeError("good-bridge selection status mismatch")
    SELECTED_SAMPLE_IDS = tuple(int(value) for value in _selection.get("sample_ids", []))
    if len(SELECTED_SAMPLE_IDS) != int(_selection.get("good_bridge_count", -1)):
        raise RuntimeError("good-bridge selection count mismatch")
else:
    _selection = None
    SELECTED_SAMPLE_IDS = tuple(range(SOURCE_BRIDGE_ROWS))
TRAIN_ROWS = len(SELECTED_SAMPLE_IDS)
GLOBAL_BATCH_SIZE = int(os.environ.get("OPD_GLOBAL_BATCH_SIZE", "64"))
STUDENT_WORLD_SIZE = int(os.environ.get("OPD_STUDENT_WORLD_SIZE", "4"))
FULL_EPOCH_STEPS = math.ceil(TRAIN_ROWS / GLOBAL_BATCH_SIZE)
TOTAL_STEPS = int(os.environ.get("OPD_TOTAL_STEPS", str(FULL_EPOCH_STEPS)))
SAVE_STEPS = tuple(int(value) for value in os.environ.get(
    "OPD_SAVE_STEPS", "331,662,993,1324,1656"
).split(","))
SEED = int(os.environ.get("OPD_SEED", "20260827"))
LOSS_SUPPORT_TOPK = 16
OBJECTIVE = "bridge_student_top_16_renormalized_kl_student_to_teacher"

if GLOBAL_BATCH_SIZE % STUDENT_WORLD_SIZE:
    raise RuntimeError("Global batch must be divisible by student world size")
if not 1 <= TOTAL_STEPS <= FULL_EPOCH_STEPS:
    raise RuntimeError("Configured total steps are outside the one-epoch range")
if SAVE_STEPS[-1] != TOTAL_STEPS:
    raise RuntimeError("Final checkpoint must be the final training step")

# Reuse the battle-tested model loading, optimizer/checkpoint, selected-logit,
# and strict final-verification machinery with bridge-specific constants.
base.TRAIN_ROWS = TRAIN_ROWS
base.GLOBAL_BATCH_SIZE = GLOBAL_BATCH_SIZE
base.STUDENT_WORLD_SIZE = STUDENT_WORLD_SIZE
base.LOCAL_BATCH_SIZE = GLOBAL_BATCH_SIZE // STUDENT_WORLD_SIZE
base.TOTAL_STEPS = TOTAL_STEPS
base.SAVE_STEPS = SAVE_STEPS
base.SEED = SEED
base.LOSS_SUPPORT_TOPK = LOSS_SUPPORT_TOPK
base.OBJECTIVE = OBJECTIVE
base.MAX_MODEL_LEN = int(os.environ.get("OPD_MAX_MODEL_LEN", "40960"))
base.HF_MAX_SHARD_SIZE = os.environ.get("OPD_HF_MAX_SHARD_SIZE", "2GB")


def load_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with BRIDGE_MANIFEST.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"bridge manifest line {line_number} is not an object")
            rows.append(row)
    rows.sort(key=lambda row: int(row["dataset_index"]))
    if len(rows) != TRACE_COUNT:
        raise RuntimeError(f"expected {TRACE_COUNT} bridge traces, found {len(rows)}")
    indices = [int(row["dataset_index"]) for row in rows]
    if len(set(indices)) != TRACE_COUNT:
        raise RuntimeError("bridge manifest dataset_index values are not unique")
    return rows


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def latest_step(checkpoint_root: Path) -> int:
    path = checkpoint_root / "latest_checkpointed_iteration.txt"
    return int(path.read_text(encoding="utf-8").strip()) if path.exists() else 0


def phase_init(args: argparse.Namespace) -> None:
    if not (BRIDGE_ROOT / "BRIDGE_BATCH_COMPLETE").is_file():
        raise FileNotFoundError(BRIDGE_ROOT / "BRIDGE_BATCH_COMPLETE")
    rows = load_manifest()
    bridge_rollouts = 0
    for row in rows:
        if int(row.get("bridge_rollout_count", -1)) != BRIDGES_PER_TRACE:
            raise RuntimeError(
                f"trace {row.get('dataset_index')}: bridge rollout count mismatch"
            )
        if int(row.get("bridge_length", -1)) != BRIDGE_TOKENS:
            raise RuntimeError(
                f"trace {row.get('dataset_index')}: bridge length mismatch"
            )
        bridge_rollouts += int(row["bridge_rollout_count"])
    if bridge_rollouts != SOURCE_BRIDGE_ROWS:
        raise RuntimeError("source bridge example accounting mismatch")
    if len(set(SELECTED_SAMPLE_IDS)) != TRAIN_ROWS:
        raise RuntimeError("selected bridge sample ids are not unique")
    if any(not 0 <= sample_id < SOURCE_BRIDGE_ROWS for sample_id in SELECTED_SAMPLE_IDS):
        raise RuntimeError("selected bridge sample id is out of range")
    if SELECTION_PATH is not None:
        if not (SELECTION_PATH.parent / "GOOD_BRIDGE_SELECTION_COMPLETE").is_file():
            raise FileNotFoundError(SELECTION_PATH.parent / "GOOD_BRIDGE_SELECTION_COMPLETE")
        if int(_selection.get("source_bridge_count", -1)) != SOURCE_BRIDGE_ROWS:
            raise RuntimeError("selection source bridge count mismatch")
        if _selection.get("bridge_manifest_sha256") != base.file_sha256(BRIDGE_MANIFEST):
            raise RuntimeError("selection bridge manifest SHA mismatch")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "student_daemon_ready.json").unlink(missing_ok=True)
    (args.run_dir / "teacher_daemons_ready.json").unlink(missing_ok=True)
    (args.run_dir / "TRAINING_DONE").unlink(missing_ok=True)
    for path in (args.run_dir / "teacher_ready").glob("rank*.ready"):
        path.unlink()
    for path in args.run_dir.glob("TEACHER_DAEMON_*_DONE"):
        path.unlink()
    checkpoint_step = latest_step(args.checkpoint_root)
    partial_steps = sorted((args.run_dir / "steps").glob("step_*"))
    uncheckpointed_metrics = [
        path
        for path in sorted((args.run_dir / "metrics").glob("step_*.json"))
        if int(path.stem.removeprefix("step_")) > checkpoint_step
    ]
    if partial_steps or uncheckpointed_metrics:
        attempt = (
            args.run_dir
            / "attempts"
            / f"job_{os.environ.get('SLURM_JOB_ID', 'manual')}_{int(time.time())}"
        )
        attempt.mkdir(parents=True, exist_ok=False)
        for path in partial_steps:
            path.rename(attempt / path.name)
        if uncheckpointed_metrics:
            target = attempt / "metrics"
            target.mkdir()
            for path in uncheckpointed_metrics:
                path.rename(target / path.name)

    order_path = args.run_dir / "train_order.json"
    if not order_path.exists():
        order = list(SELECTED_SAMPLE_IDS)
        random.Random(SEED).shuffle(order)
        base.atomic_json(order_path, {"seed": SEED, "indices": order})
    order = read_json(order_path)
    if (
        order.get("seed") != SEED
        or len(order.get("indices", [])) != TRAIN_ROWS
        or len(set(order["indices"])) != TRAIN_ROWS
        or set(int(value) for value in order["indices"]) != set(SELECTED_SAMPLE_IDS)
    ):
        raise RuntimeError("training order is not the expected permutation")

    run_manifest = {
        "status": "BRIDGE_TOP16_RKL_INITIALIZED",
        "objective": OBJECTIVE,
        "loss_direction": "KL(student||teacher)",
        "loss_support_topk": LOSS_SUPPORT_TOPK,
        "loss_support_selection": (
            "current student logits; stop-gradient; teacher and student "
            "renormalized over identical selected support"
        ),
        "loss_mask": "bridge_tokens_only",
        "conditioning": "anchor_prefix_token_ids + bridge teacher forcing",
        "student_model": str(base.STUDENT_MODEL),
        "teacher_model": str(base.TEACHER_MODEL),
        "bridge_root": str(BRIDGE_ROOT),
        "bridge_manifest": str(BRIDGE_MANIFEST),
        "bridge_manifest_sha256": base.file_sha256(BRIDGE_MANIFEST),
        "bridge_completion_marker": str(BRIDGE_ROOT / "BRIDGE_BATCH_COMPLETE"),
        "trace_count": TRACE_COUNT,
        "bridges_per_trace": BRIDGES_PER_TRACE,
        "bridge_tokens": BRIDGE_TOKENS,
        "train_rows": TRAIN_ROWS,
        "source_bridge_rows": SOURCE_BRIDGE_ROWS,
        "training_scope": "good_bridges_only" if SELECTION_PATH is not None else "all_bridges",
        "bridge_selection": str(SELECTION_PATH) if SELECTION_PATH is not None else None,
        "bridge_selection_sha256": base.file_sha256(SELECTION_PATH) if SELECTION_PATH is not None else None,
        "bridge_selection_status": _selection.get("status") if _selection is not None else None,
        "bridge_selection_definition": _selection.get("definition") if _selection is not None else None,
        "epochs": 1 if TOTAL_STEPS == FULL_EPOCH_STEPS else None,
        "seed": SEED,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "student_world_size": STUDENT_WORLD_SIZE,
        "max_local_batch_size": GLOBAL_BATCH_SIZE // STUDENT_WORLD_SIZE,
        "total_steps": TOTAL_STEPS,
        "save_steps": list(SAVE_STEPS),
        "max_model_len": base.MAX_MODEL_LEN,
        "model_vocab_size": base.MODEL_VOCAB_SIZE,
        "learning_rate": 1e-6,
        "hf_max_shard_size": base.HF_MAX_SHARD_SIZE,
        "teacher_daemon_count": STUDENT_WORLD_SIZE,
        "all_bridges_covered_once": SELECTION_PATH is None and TOTAL_STEPS == FULL_EPOCH_STEPS,
        "selected_bridges_covered_once": TOTAL_STEPS == FULL_EPOCH_STEPS,
    }
    base.atomic_json(args.run_dir / "run_manifest.json", run_manifest)
    print("BRIDGE_TOP16_RKL_INIT_OK", json.dumps(run_manifest, sort_keys=True))


def step_sample_ids(run_dir: Path, step: int) -> list[int]:
    if not 1 <= step <= TOTAL_STEPS:
        raise ValueError(step)
    order = read_json(run_dir / "train_order.json")["indices"]
    start = (step - 1) * GLOBAL_BATCH_SIZE
    return [int(value) for value in order[start : min(start + GLOBAL_BATCH_SIZE, TRAIN_ROWS)]]


def split_for_ranks(values: list[int]) -> list[list[int]]:
    if len(values) % STUDENT_WORLD_SIZE:
        raise RuntimeError("step batch cannot be divided evenly across student ranks")
    width = len(values) // STUDENT_WORLD_SIZE
    return [values[rank * width : (rank + 1) * width] for rank in range(STUDENT_WORLD_SIZE)]


def phase_prepare_step(args: argparse.Namespace) -> None:
    manifest = load_manifest()
    sample_ids = step_sample_ids(args.run_dir, args.step)
    rank_samples = split_for_ranks(sample_ids)
    if args.step_dir.exists():
        raise FileExistsError(args.step_dir)
    rows_compact: list[dict[str, Any]] = []
    for rank, local_ids in enumerate(rank_samples):
        rank_dir = args.step_dir / "traces" / f"rank{rank}"
        rank_dir.mkdir(parents=True, exist_ok=False)
        compact: list[dict[str, Any]] = []
        for local_index, sample_id in enumerate(local_ids):
            manifest_row = manifest[sample_id // BRIDGES_PER_TRACE]
            rollout_index = sample_id % BRIDGES_PER_TRACE
            anchor_path = Path(manifest_row["anchor_artifact"])
            bridge_path = Path(manifest_row["bridge_artifact"])
            anchor = read_json(anchor_path)
            bridge_bundle = read_json(bridge_path)
            bridge_rollouts = bridge_bundle.get("rollouts")
            if not isinstance(bridge_rollouts, list) or len(bridge_rollouts) != BRIDGES_PER_TRACE:
                raise RuntimeError(f"sample {sample_id}: invalid bridge bundle")
            prefix_ids = [int(value) for value in anchor["anchor_prefix_token_ids"]]
            bridge_ids = [
                int(value) for value in bridge_rollouts[rollout_index]["bridge_token_ids"]
            ]
            if len(bridge_ids) != BRIDGE_TOKENS:
                raise RuntimeError(f"sample {sample_id}: invalid bridge length")
            if bridge_ids[0] == int(anchor["student_candidate_token_id"]):
                raise RuntimeError(f"sample {sample_id}: bridge starts with student anchor token")
            if len(prefix_ids) + len(bridge_ids) > base.MAX_MODEL_LEN:
                raise RuntimeError(f"sample {sample_id}: sequence exceeds context")
            trace = {
                "status": "BRIDGE_TEACHER_FORCING_TRACE_OK",
                "step": args.step,
                "rank": rank,
                "local_index": local_index,
                "sample_id": sample_id,
                "dataset_index": int(manifest_row["dataset_index"]),
                "problem_id": manifest_row.get("problem_id"),
                "rollout_index": rollout_index,
                "anchor_artifact": str(anchor_path),
                "bridge_artifact": str(bridge_path),
                "prompt_token_ids": prefix_ids,
                "response_token_ids": bridge_ids,
                "prompt_tokens": len(prefix_ids),
                "response_tokens": len(bridge_ids),
                "loss_mask_start": len(prefix_ids),
                "loss_mask_tokens": BRIDGE_TOKENS,
                "loss_support_topk": LOSS_SUPPORT_TOPK,
                "student_topk_file": f"sample{local_index:02d}.student_topk.pt",
            }
            base.atomic_json(rank_dir / f"sample{local_index:02d}.json", trace)
            row = {
                "rank": rank,
                "local_index": local_index,
                "sample_id": sample_id,
                "dataset_index": trace["dataset_index"],
                "rollout_index": rollout_index,
                "prompt_tokens": len(prefix_ids),
                "response_tokens": len(bridge_ids),
            }
            compact.append(row)
            rows_compact.append(row)
        base.atomic_json(rank_dir / "manifest.json", {
            "status": "ROLLOUT_RANK_OK",
            "step": args.step,
            "rank": rank,
            "examples": len(compact),
            "rows": compact,
        })
    if len({row["sample_id"] for row in rows_compact}) != len(sample_ids):
        raise RuntimeError("duplicate sample within training step")
    merged = {
        "status": "ROLLOUT_GLOBAL_OK",
        "step": args.step,
        "examples": len(rows_compact),
        "total_response_tokens": len(rows_compact) * BRIDGE_TOKENS,
        "max_response_tokens_observed": BRIDGE_TOKENS,
        "min_effective_max_response_tokens": BRIDGE_TOKENS,
        "generated_thinking_tag_count": 0,
        "finish_reasons": {"teacher_forcing": len(rows_compact)},
        "sample_id_min": min(sample_ids),
        "sample_id_max": max(sample_ids),
        "rows": rows_compact,
    }
    base.atomic_json(args.step_dir / "rollout_manifest.json", merged)
    (args.step_dir / "ROLLOUT_READY").touch()
    print("BRIDGE_TOP16_STEP_PREPARED", json.dumps(merged, sort_keys=True))


def compute_student_support(model: Any, trace: dict[str, Any], output: Path, torch: Any) -> None:
    values = trace["prompt_token_ids"] + trace["response_token_ids"]
    input_ids = torch.tensor(values, dtype=torch.long, device="cuda").unsqueeze(0)
    prompt_tokens = int(trace["prompt_tokens"])
    response_tokens = int(trace["response_tokens"])
    selected_chunks = []
    was_training = model.training
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        position0 = prompt_tokens - 1
        for start in range(0, response_tokens, base.CHUNK_TOKENS):
            end = min(start + base.CHUNK_TOKENS, response_tokens)
            logits = model.lm_head(hidden[:, position0 + start : position0 + end]).squeeze(0)
            if logits.shape != (end - start, base.MODEL_VOCAB_SIZE):
                raise RuntimeError(f"student support logits shape mismatch: {logits.shape}")
            selected_chunks.append(
                torch.topk(logits.float(), k=LOSS_SUPPORT_TOPK, dim=-1).indices
                .to(dtype=torch.int32, device="cpu")
            )
            del logits
    if was_training:
        model.train()
    selected = torch.cat(selected_chunks, dim=0).contiguous()
    if selected.shape != (response_tokens, LOSS_SUPPORT_TOPK):
        raise RuntimeError(f"student top-k shape mismatch: {selected.shape}")
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(selected, temporary)
    temporary.replace(output)
    del selected, selected_chunks, hidden, input_ids


def phase_teacher_daemon(args: argparse.Namespace) -> None:
    import torch

    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = base.load_lm(base.TEACHER_MODEL, torch.bfloat16, training=False)
    ready_dir = args.run_dir / "teacher_ready"
    ready_dir.mkdir(parents=True, exist_ok=True)
    (ready_dir / f"rank{args.rank}.ready").touch()
    start_step = latest_step(args.checkpoint_root) + 1
    for step in range(start_step, TOTAL_STEPS + 1):
        step_dir = args.run_dir / "steps" / f"step_{step:04d}"
        base.wait_for(step_dir / "STUDENT_SUPPORT_READY", timeout_seconds=259200)
        trace_manifest = read_json(step_dir / "traces" / f"rank{args.rank}" / "manifest.json")
        completed = []
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        for local_index in range(int(trace_manifest["examples"])):
            trace = read_json(step_dir / "traces" / f"rank{args.rank}" / f"sample{local_index:02d}.json")
            selected_ids = torch.load(
                step_dir / "traces" / f"rank{args.rank}" / trace["student_topk_file"],
                map_location="cpu", weights_only=True,
            ).long()
            if selected_ids.shape != (BRIDGE_TOKENS, LOSS_SUPPORT_TOPK):
                raise RuntimeError("teacher selected support shape mismatch")
            values = trace["prompt_token_ids"] + trace["response_token_ids"]
            input_ids = torch.tensor(values, dtype=torch.long, device="cuda").unsqueeze(0)
            output_dir = step_dir / "teacher" / f"rank{args.rank}" / f"sample{local_index:02d}"
            output_dir.mkdir(parents=True, exist_ok=False)
            chunks = []
            sample_started = time.monotonic()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = model.model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state
                position0 = int(trace["prompt_tokens"]) - 1
                for chunk_index, start in enumerate(range(0, BRIDGE_TOKENS, base.CHUNK_TOKENS)):
                    end = min(start + base.CHUNK_TOKENS, BRIDGE_TOKENS)
                    logits = base.selected_lm_logits(
                        model,
                        hidden[:, position0 + start : position0 + end],
                        selected_ids[start:end].to(device="cuda", non_blocking=True),
                        torch,
                    )
                    if logits.shape != (end - start, LOSS_SUPPORT_TOPK):
                        raise RuntimeError("teacher selected logits shape mismatch")
                    cpu_logits = logits.to(dtype=torch.bfloat16, device="cpu").contiguous()
                    path = output_dir / f"chunk_{chunk_index:04d}.pt"
                    temporary = path.with_suffix(".pt.tmp")
                    torch.save(cpu_logits, temporary)
                    temporary.replace(path)
                    chunks.append({
                        "index": chunk_index,
                        "start": start,
                        "end": end,
                        "tokens": end - start,
                        "shape": list(cpu_logits.shape),
                        "bytes": path.stat().st_size,
                        "file": path.name,
                    })
                    del logits, cpu_logits
            sample_manifest = {
                "status": "TEACHER_RKL_SUPPORT_OK",
                "step": step,
                "rank": args.rank,
                "local_index": local_index,
                "sample_id": trace["sample_id"],
                "dataset_index": trace["dataset_index"],
                "rollout_index": trace["rollout_index"],
                "response_tokens": BRIDGE_TOKENS,
                "loss_support_topk": LOSS_SUPPORT_TOPK,
                "chunks": chunks,
                "elapsed_seconds": time.monotonic() - sample_started,
                "bytes": sum(chunk["bytes"] for chunk in chunks),
            }
            base.atomic_json(output_dir / "manifest.json", sample_manifest)
            completed.append({
                "local_index": local_index,
                "sample_id": trace["sample_id"],
                "response_tokens": BRIDGE_TOKENS,
                "elapsed_seconds": sample_manifest["elapsed_seconds"],
                "bytes": sample_manifest["bytes"],
            })
            del input_ids, hidden, selected_ids
            gc.collect()
            torch.cuda.empty_cache()
        rank_manifest = {
            "status": "TEACHER_RANK_OK",
            "step": step,
            "rank": args.rank,
            "examples": len(completed),
            "elapsed_seconds": time.monotonic() - started,
            "cuda": base.cuda_metrics(torch),
            "rows": completed,
        }
        base.atomic_json(step_dir / "teacher" / f"rank{args.rank}" / "manifest.json", rank_manifest)
        (step_dir / f"TEACHER_RANK_{args.rank}_DONE").touch()
        print("BRIDGE_TOP16_TEACHER_STEP_OK", json.dumps(rank_manifest, sort_keys=True), flush=True)
    # Keep the persistent teacher process alive until the final student update
    # is committed. Otherwise the parent could mistake a successful early
    # teacher exit for a daemon failure while the student consumes final logits.
    base.wait_for(args.run_dir / "TRAINING_DONE", timeout_seconds=259200)
    (args.run_dir / f"TEACHER_DAEMON_{args.rank}_DONE").touch()


def phase_student_daemon(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from datetime import timedelta

    dist.init_process_group("nccl", timeout=timedelta(hours=12))
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if dist.get_world_size() != STUDENT_WORLD_SIZE:
        raise RuntimeError("student world-size mismatch")
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    checkpoint_step, model_path = base.latest_checkpoint(args.checkpoint_root)
    model = base.load_lm(model_path, torch.float32, training=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-6, weight_decay=0.01, betas=(0.9, 0.999)
    )
    if checkpoint_step:
        actor = args.checkpoint_root / f"global_step_{checkpoint_step}" / "actor"
        optimizer.load_state_dict(torch.load(actor / "optimizer.pt", map_location="cuda", weights_only=False))
        extra = torch.load(actor / "extra_state.pt", map_location="cpu", weights_only=False)
        torch.set_rng_state(extra["torch_rng_state"])
        torch.cuda.set_rng_state(extra["cuda_rng_state"].to("cpu"))
    if rank == 0:
        args.checkpoint_root.mkdir(parents=True, exist_ok=True)
        base.export_current_model(model, args.run_dir)
        base.atomic_json(args.run_dir / "student_daemon_ready.json", {
            "status": "STUDENT_DAEMON_READY",
            "checkpoint_step": checkpoint_step,
            "world_size": STUDENT_WORLD_SIZE,
            "model_path": str(model_path),
        })
    dist.barrier()

    for step in range(checkpoint_step + 1, TOTAL_STEPS + 1):
        step_dir = args.run_dir / "steps" / f"step_{step:04d}"
        base.wait_for(step_dir / "ROLLOUT_READY", timeout_seconds=259200)
        rollout = read_json(step_dir / "rollout_manifest.json")
        global_tokens = int(rollout["total_response_tokens"])
        trace_manifest = read_json(step_dir / "traces" / f"rank{rank}" / "manifest.json")
        local_examples = int(trace_manifest["examples"])
        if global_tokens != int(rollout["examples"]) * BRIDGE_TOKENS:
            raise RuntimeError("bridge token accounting mismatch")

        for local_index in range(local_examples):
            trace = read_json(step_dir / "traces" / f"rank{rank}" / f"sample{local_index:02d}.json")
            compute_student_support(
                model,
                trace,
                step_dir / "traces" / f"rank{rank}" / trace["student_topk_file"],
                torch,
            )
            gc.collect()
            torch.cuda.empty_cache()
        (step_dir / f"STUDENT_SUPPORT_RANK_{rank}_DONE").touch()
        dist.barrier()
        if rank == 0:
            (step_dir / "STUDENT_SUPPORT_READY").touch()
        dist.barrier()

        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        checksum_before = base.parameter_checksum(model, torch)
        local_loss_sum = 0.0
        local_tokens = 0
        sample_rows = []
        for local_index in range(local_examples):
            trace = read_json(step_dir / "traces" / f"rank{rank}" / f"sample{local_index:02d}.json")
            teacher_dir = step_dir / "teacher" / f"rank{rank}" / f"sample{local_index:02d}"
            base.wait_for(teacher_dir / "manifest.json", timeout_seconds=259200)
            teacher = read_json(teacher_dir / "manifest.json")
            selected_ids = torch.load(
                step_dir / "traces" / f"rank{rank}" / trace["student_topk_file"],
                map_location="cpu", weights_only=True,
            ).long()
            values = trace["prompt_token_ids"] + trace["response_token_ids"]
            input_ids = torch.tensor(values, dtype=torch.long, device="cuda").unsqueeze(0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = model.model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state
            hidden_gradient = torch.zeros_like(hidden)
            position0 = int(trace["prompt_tokens"]) - 1
            sample_loss = 0.0
            for chunk in teacher["chunks"]:
                start, end = int(chunk["start"]), int(chunk["end"])
                teacher_logits = torch.load(
                    teacher_dir / chunk["file"], map_location="cpu", weights_only=True
                ).to(device="cuda", dtype=torch.bfloat16)
                chunk_hidden = hidden[:, position0 + start : position0 + end].detach().requires_grad_(True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    student_logits = base.selected_lm_logits(
                        model,
                        chunk_hidden,
                        selected_ids[start:end].to(device="cuda", non_blocking=True),
                        torch,
                    )
                student_logp = F.log_softmax(student_logits.float(), dim=-1)
                teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
                probabilities = student_logp.exp()
                token_kl = (probabilities * (student_logp - teacher_logp)).sum(dim=-1)
                if not torch.isfinite(token_kl).all():
                    raise FloatingPointError("non-finite bridge reverse KL")
                chunk_sum = token_kl.sum()
                (chunk_sum / global_tokens).backward()
                hidden_gradient[:, position0 + start : position0 + end].copy_(chunk_hidden.grad)
                sample_loss += float(chunk_sum.detach().cpu())
                del teacher_logits, chunk_hidden, student_logits, student_logp
                del teacher_logp, probabilities, token_kl, chunk_sum
            hidden.backward(hidden_gradient)
            local_loss_sum += sample_loss
            local_tokens += BRIDGE_TOKENS
            sample_rows.append({
                "local_index": local_index,
                "sample_id": trace["sample_id"],
                "dataset_index": trace["dataset_index"],
                "rollout_index": trace["rollout_index"],
                "bridge_tokens": BRIDGE_TOKENS,
                "mean_reverse_kl": sample_loss / BRIDGE_TOKENS,
            })
            del input_ids, hidden, hidden_gradient, selected_ids
            shutil.rmtree(teacher_dir)
            gc.collect()
            torch.cuda.empty_cache()

        for parameter in model.parameters():
            if parameter.grad is None:
                raise RuntimeError("trainable parameter missing gradient")
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu())
        if not math.isfinite(grad_norm):
            raise FloatingPointError("non-finite gradient norm")
        optimizer.step()
        checksum_after = base.parameter_checksum(model, torch)
        local_metrics = {
            "rank": rank,
            "local_examples": local_examples,
            "local_tokens": local_tokens,
            "local_loss_sum": local_loss_sum,
            "elapsed_seconds": time.monotonic() - started,
            "cuda": base.cuda_metrics(torch),
            "samples": sample_rows,
        }
        gathered: list[Any] | None = [None] * STUDENT_WORLD_SIZE if rank == 0 else None
        dist.gather_object(local_metrics, gathered, dst=0)
        if rank == 0:
            assert gathered is not None
            metrics = {
                "status": "BRIDGE_TOP16_RKL_STEP_OK",
                "step": step,
                "objective": OBJECTIVE,
                "loss_support_topk": LOSS_SUPPORT_TOPK,
                "loss_mask": "bridge_tokens_only",
                "global_batch_size_actual": int(rollout["examples"]),
                "global_bridge_tokens": global_tokens,
                "mean_reverse_kl": sum(row["local_loss_sum"] for row in gathered) / global_tokens,
                "gradient_norm_before_clip": grad_norm,
                "parameter_checksum_before": checksum_before,
                "parameter_checksum_after": checksum_after,
                "parameter_checksum_delta": checksum_after - checksum_before,
                "rank_metrics": gathered,
            }
            if metrics["parameter_checksum_delta"] == 0.0:
                raise RuntimeError("optimizer step did not change parameters")
            base.atomic_json(step_dir / "step_metrics.json", metrics)
            base.export_current_model(model, args.run_dir)
            if step in SAVE_STEPS:
                base.save_checkpoint(model, optimizer, args.checkpoint_root, args.run_dir, step, metrics, torch)
            (step_dir / "STUDENT_DONE").touch()
            print("BRIDGE_TOP16_RKL_STEP_OK", json.dumps(metrics, sort_keys=True), flush=True)
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
    if rank == 0:
        (args.run_dir / "TRAINING_DONE").touch()
    dist.barrier()
    dist.destroy_process_group()


def phase_step_summary(args: argparse.Namespace) -> None:
    base.wait_for(args.step_dir / "STUDENT_DONE", timeout_seconds=259200)
    student = read_json(args.step_dir / "step_metrics.json")
    rollout = read_json(args.step_dir / "rollout_manifest.json")
    teachers = [
        read_json(args.step_dir / "teacher" / f"rank{rank}" / "manifest.json")
        for rank in range(STUDENT_WORLD_SIZE)
    ]
    if any(item.get("status") != "TEACHER_RANK_OK" for item in teachers):
        raise RuntimeError("teacher rank incomplete")
    summary = {
        "status": "BRIDGE_TOP16_RKL_STEP_COMPLETE",
        "step": args.step,
        "rollout": rollout,
        "student": student,
        "teacher_elapsed_max_seconds": max(item["elapsed_seconds"] for item in teachers),
        "teacher_peak_allocated_max_gib": max(item["cuda"]["peak_allocated_gib"] for item in teachers),
    }
    base.atomic_json(args.run_dir / "metrics" / f"step_{args.step:04d}.json", summary)
    shutil.rmtree(args.step_dir)
    print("BRIDGE_TOP16_RKL_STEP_COMPLETE", json.dumps(summary, sort_keys=True))


def phase_verify(args: argparse.Namespace) -> None:
    base.phase_verify(args)
    result_path = args.run_dir / "final_result.json"
    result = read_json(result_path)
    result.update({
        "status": "BRIDGE_TOP16_RKL_COMPLETE",
        "objective": OBJECTIVE,
        "loss_mask": "bridge_tokens_only",
        "train_rows": min(TRAIN_ROWS, TOTAL_STEPS * GLOBAL_BATCH_SIZE),
        "trace_count": TRACE_COUNT,
        "bridges_per_trace": BRIDGES_PER_TRACE,
        "bridge_tokens": BRIDGE_TOKENS,
        "epochs": 1 if TOTAL_STEPS == FULL_EPOCH_STEPS else None,
        "all_bridges_covered_once": SELECTION_PATH is None and TOTAL_STEPS == FULL_EPOCH_STEPS,
        "selected_bridges_covered_once": TOTAL_STEPS == FULL_EPOCH_STEPS,
        "training_scope": "good_bridges_only" if SELECTION_PATH is not None else "all_bridges",
        "bridge_selection": str(SELECTION_PATH) if SELECTION_PATH is not None else None,
        "bridge_selection_sha256": base.file_sha256(SELECTION_PATH) if SELECTION_PATH is not None else None,
    })
    result.pop("thinking", None)
    base.atomic_json(result_path, result)
    print("BRIDGE_TOP16_RKL_COMPLETE", json.dumps(result, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=(
        "init", "prepare-step", "teacher-daemon", "student-daemon",
        "step-summary", "verify",
    ))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()
    args.step_dir = args.run_dir / "steps" / f"step_{args.step:04d}"
    return args


def main() -> None:
    args = parse_args()
    {
        "init": phase_init,
        "prepare-step": phase_prepare_step,
        "teacher-daemon": phase_teacher_daemon,
        "student-daemon": phase_student_daemon,
        "step-summary": phase_step_summary,
        "verify": phase_verify,
    }[args.phase](args)


if __name__ == "__main__":
    main()
