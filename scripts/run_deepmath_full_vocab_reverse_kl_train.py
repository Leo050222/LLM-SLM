#!/usr/bin/env python3
"""Exact full-model-vocabulary on-policy Reverse-KL training.

Four persistent student ranks hold Qwen3-1.7B and AdamW state.  Four teacher
workers hold Qwen3-14B while scoring each step.  Student rollouts are generated
from an atomically exported snapshot of the current policy.  For every sampled
prefix, KL(student || teacher) is summed over all 151,936 model-head logits.
Token-position chunks bound peak memory; the vocabulary is never truncated.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import time
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any


# Concurrent vLLM/Triton processes must not share the default cache's temporary
# directory.  A process-local cache prevents atomic-rename/removedirs races
# without changing kernels, model numerics, sampling, or training semantics.
if os.environ.get("SLURM_JOB_ID"):
    os.environ.setdefault(
        "TRITON_CACHE_DIR",
        f"/tmp/opd_triton_{os.environ['SLURM_JOB_ID']}_{os.getpid()}",
    )


STUDENT_MODEL = Path("/storage/yukaichengLab/share/model/Qwen3-1.7B")
TEACHER_MODEL = Path("/storage/yukaichengLab/share/model/Qwen3-14B")
TRAIN_FILE = Path(os.environ.get(
    "OPD_TRAIN_FILE",
    "/storage/yukaichengLab/wangxin/test_eiloeo/opd_data/deepmath_ge6_opd/train.parquet",
))
DATA_MANIFEST = TRAIN_FILE.parent / "manifest.json"
SEED = int(os.environ.get("OPD_SEED", "20260810"))
TRAIN_ROWS = int(os.environ.get("OPD_TRAIN_ROWS", "56640"))
GLOBAL_BATCH_SIZE = int(os.environ.get("OPD_GLOBAL_BATCH_SIZE", "64"))
STUDENT_WORLD_SIZE = int(os.environ.get("OPD_STUDENT_WORLD_SIZE", "4"))
LOCAL_BATCH_SIZE = GLOBAL_BATCH_SIZE // STUDENT_WORLD_SIZE
TOTAL_STEPS = int(os.environ.get("OPD_TOTAL_STEPS", "885"))
SAVE_STEPS = tuple(int(value) for value in os.environ.get(
    "OPD_SAVE_STEPS", "177,354,531,708,885"
).split(","))
MAX_MODEL_LEN = int(os.environ.get("OPD_MAX_MODEL_LEN", "40960"))
MAX_PROMPT_TOKENS = int(os.environ.get("OPD_MAX_PROMPT_TOKENS", "695"))
MAX_RESPONSE_TOKENS = int(os.environ.get("OPD_MAX_RESPONSE_TOKENS", "40264"))
HF_MAX_SHARD_SIZE = os.environ.get("OPD_HF_MAX_SHARD_SIZE", "2GB")
ENABLE_THINKING = os.environ.get("OPD_ENABLE_THINKING", "1").strip().lower() in {
    "1", "true", "yes", "on",
}
TOKENIZER_VOCAB_SIZE = 151669
MODEL_VOCAB_SIZE = 151936
CHUNK_TOKENS = 256
LOSS_SUPPORT_TOPK = int(os.environ.get("OPD_LOSS_SUPPORT_TOPK", "0"))
OBJECTIVE = (
    f"student_top_{LOSS_SUPPORT_TOPK}_renormalized_kl_student_to_teacher"
    if LOSS_SUPPORT_TOPK
    else "exact_full_model_vocabulary_kl_student_to_teacher"
)
DATA_STATUS = os.environ.get("OPD_DATA_STATUS", "DEEPMATH_GE6_OPD_DATA_OK")
EXPECTED_DATA_TRAIN_STEPS = int(os.environ.get(
    "OPD_EXPECTED_DATA_TRAIN_STEPS", str(TOTAL_STEPS)
))
EXPECTED_TRAIN_SHA256 = os.environ.get("OPD_TRAIN_SHA256", "")

if GLOBAL_BATCH_SIZE % STUDENT_WORLD_SIZE:
    raise RuntimeError("Global batch must be divisible by student world size")
if LOSS_SUPPORT_TOPK < 0 or LOSS_SUPPORT_TOPK > MODEL_VOCAB_SIZE:
    raise RuntimeError(f"Invalid loss support top-k: {LOSS_SUPPORT_TOPK}")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def wait_for(path: Path, timeout_seconds: float = 86400) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.exists():
            return
        if (path.parent / "FAILED").exists():
            raise RuntimeError(f"Upstream phase failed while waiting for {path}")
        time.sleep(2)
    raise TimeoutError(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cuda_metrics(torch: Any) -> dict[str, float]:
    return {
        "allocated_gib": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gib": torch.cuda.memory_reserved() / 1024**3,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }


def load_rows() -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows = pq.read_table(TRAIN_FILE).to_pylist()
    if len(rows) != TRAIN_ROWS:
        raise RuntimeError(f"Expected {TRAIN_ROWS} train rows, found {len(rows)}")
    return rows


def phase_init(args: argparse.Namespace) -> None:
    manifest = read_json(DATA_MANIFEST)
    expected = {
        "status": DATA_STATUS,
        "train_total_rows": TRAIN_ROWS,
        "train_steps": EXPECTED_DATA_TRAIN_STEPS,
        "test_rows": 1000,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Manifest mismatch {key}: {manifest.get(key)} != {value}")
    actual_train_sha256 = file_sha256(TRAIN_FILE)
    if EXPECTED_TRAIN_SHA256 and actual_train_sha256 != EXPECTED_TRAIN_SHA256:
        raise RuntimeError(
            f"Train SHA mismatch: {actual_train_sha256} != {EXPECTED_TRAIN_SHA256}"
        )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    # Readiness files and partial step directories are attempt-local. Preserve
    # failed evidence under attempts/ instead of overwriting or deleting it.
    (args.run_dir / "student_daemon_ready.json").unlink(missing_ok=True)
    checkpoint_step = 0
    checkpoint_root_text = os.environ.get("FULL_RKL_CHECKPOINT_ROOT")
    if checkpoint_root_text:
        latest = Path(checkpoint_root_text) / "latest_checkpointed_iteration.txt"
        if latest.exists():
            checkpoint_step = int(latest.read_text(encoding="utf-8").strip())
    steps_dir = args.run_dir / "steps"
    partial_steps = sorted(steps_dir.glob("step_*")) if steps_dir.exists() else []
    metrics_dir = args.run_dir / "metrics"
    uncheckpointed_metrics = []
    if metrics_dir.exists():
        for path in sorted(metrics_dir.glob("step_*.json")):
            metric_step = int(path.stem.removeprefix("step_"))
            if metric_step > checkpoint_step:
                uncheckpointed_metrics.append(path)
    if partial_steps or uncheckpointed_metrics:
        attempt_dir = (
            args.run_dir / "attempts" /
            f"job_{os.environ.get('SLURM_JOB_ID', 'manual')}_{int(time.time())}"
        )
        attempt_dir.mkdir(parents=True, exist_ok=False)
        for path in partial_steps:
            path.rename(attempt_dir / path.name)
        if uncheckpointed_metrics:
            archived_metrics = attempt_dir / "metrics"
            archived_metrics.mkdir()
            for path in uncheckpointed_metrics:
                path.rename(archived_metrics / path.name)
    order_path = args.run_dir / "train_order.json"
    if not order_path.exists():
        order = list(range(TRAIN_ROWS))
        random.Random(SEED).shuffle(order)
        atomic_json(order_path, {"seed": SEED, "indices": order})
    order = read_json(order_path)
    if len(order["indices"]) != TRAIN_ROWS or len(set(order["indices"])) != TRAIN_ROWS:
        raise RuntimeError("Training order is not a permutation")
    run_manifest = {
        "status": "DEEPMATH_RKL_INITIALIZED",
        "objective": OBJECTIVE,
        "student_model": str(STUDENT_MODEL),
        "teacher_model": str(TEACHER_MODEL),
        "train_file": str(TRAIN_FILE),
        "train_file_sha256": actual_train_sha256,
        "data_manifest_sha256": file_sha256(DATA_MANIFEST),
        "order_sha256": file_sha256(order_path),
        "seed": SEED,
        "train_rows": TRAIN_ROWS,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "student_world_size": STUDENT_WORLD_SIZE,
        "local_batch_size": LOCAL_BATCH_SIZE,
        "total_steps": TOTAL_STEPS,
        "save_steps": list(SAVE_STEPS),
        "model_vocab_size": MODEL_VOCAB_SIZE,
        "tokenizer_vocab_size": TOKENIZER_VOCAB_SIZE,
        "loss_support_topk": LOSS_SUPPORT_TOPK,
        "loss_support_selection": (
            "student rollout distribution; stop-gradient; renormalized over selected support"
            if LOSS_SUPPORT_TOPK else "full model vocabulary"
        ),
        "chunk_tokens": CHUNK_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "max_response_tokens": MAX_RESPONSE_TOKENS,
        "hf_max_shard_size": HF_MAX_SHARD_SIZE,
        "thinking": ENABLE_THINKING,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "learning_rate": 1e-6,
        "checkpoint_contents": ["huggingface", "model", "optimizer", "extra_state"],
    }
    atomic_json(args.run_dir / "run_manifest.json", run_manifest)
    print("DEEPMATH_RKL_INIT_OK", json.dumps(run_manifest, sort_keys=True))


def batch_indices(run_dir: Path, step: int, rank: int) -> list[int]:
    if not 1 <= step <= TOTAL_STEPS or not 0 <= rank < STUDENT_WORLD_SIZE:
        raise ValueError((step, rank))
    order = read_json(run_dir / "train_order.json")["indices"]
    batch_start = (step - 1) * GLOBAL_BATCH_SIZE
    local_start = batch_start + rank * LOCAL_BATCH_SIZE
    return [int(value) for value in order[local_start : local_start + LOCAL_BATCH_SIZE]]


def rollout_topk_ids(result: Any) -> list[list[int]]:
    """Return the student's unmodified-distribution top-k at every sampled prefix."""
    if not LOSS_SUPPORT_TOPK:
        return []
    if result.logprobs is None or len(result.logprobs) != len(result.token_ids):
        raise RuntimeError("vLLM did not return one logprob row per response token")
    rows: list[list[int]] = []
    for position, values in enumerate(result.logprobs):
        candidates: list[tuple[float, int]] = []
        for token_id, value in values.items():
            token = int(token_id)
            if not 0 <= token < MODEL_VOCAB_SIZE:
                continue
            logprob = float(value.logprob if hasattr(value, "logprob") else value)
            if math.isfinite(logprob):
                candidates.append((logprob, token))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        selected = [token for _, token in candidates[:LOSS_SUPPORT_TOPK]]
        if len(selected) != LOSS_SUPPORT_TOPK or len(set(selected)) != LOSS_SUPPORT_TOPK:
            raise RuntimeError(
                f"Invalid student top-k row at position {position}: {len(selected)}"
            )
        rows.append(selected)
    return rows


def phase_rollout(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    rows = load_rows()
    indices = batch_indices(args.run_dir, args.step, args.rank)
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, local_files_only=True)
    prompts: list[list[int]] = []
    for index in indices:
        encoded = tokenizer.apply_chat_template(
            rows[index]["prompt"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING,
            return_dict=True,
        )
        ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        ids = [int(value) for value in ids]
        if len(ids) > MAX_PROMPT_TOKENS:
            raise RuntimeError(f"Prompt {index} has {len(ids)} tokens")
        prompts.append(ids)

    started = time.monotonic()
    llm = LLM(
        model=str(args.model_dir),
        tokenizer=str(STUDENT_MODEL),
        tensor_parallel_size=1,
        dtype="bfloat16",
        seed=SEED,
        gpu_memory_utilization=0.30,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=4,
        enable_prefix_caching=False,
        enforce_eager=True,
        disable_log_stats=True,
    )
    # The formal launcher disables V1 EngineCore multiprocessing.  Keeping the
    # engine in this rank process avoids an otherwise silent parent/child death
    # observed under four concurrent long-context rollouts.  Delay touching
    # torch CUDA state until after vLLM has initialized the device.
    torch.cuda.reset_peak_memory_stats()
    sampling_kwargs = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "n": 1,
    }
    if LOSS_SUPPORT_TOPK:
        sampling_kwargs["logprobs"] = LOSS_SUPPORT_TOPK
    sampling = [
        SamplingParams(
            **sampling_kwargs,
            max_tokens=min(MAX_RESPONSE_TOKENS, MAX_MODEL_LEN - len(prompt_ids)),
            seed=SEED + (args.step - 1) * TRAIN_ROWS + index,
        )
        for index, prompt_ids in zip(indices, prompts, strict=True)
    ]
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=ids) for ids in prompts],
        sampling,
        use_tqdm=True,
    )
    rank_dir = args.step_dir / "traces" / f"rank{args.rank}"
    rank_dir.mkdir(parents=True, exist_ok=False)
    trace_rows: list[dict[str, Any]] = []
    for local_index, (dataset_index, prompt_ids, request) in enumerate(
        zip(indices, prompts, outputs, strict=True)
    ):
        result = request.outputs[0]
        response_ids = [int(value) for value in result.token_ids]
        if not response_ids:
            raise RuntimeError(f"Empty response at dataset index {dataset_index}")
        if len(prompt_ids) + len(response_ids) > MAX_MODEL_LEN:
            raise RuntimeError("Trace exceeds context")
        effective_max_response_tokens = min(
            MAX_RESPONSE_TOKENS, MAX_MODEL_LEN - len(prompt_ids)
        )
        decoded_response = tokenizer.decode(response_ids, skip_special_tokens=False)
        generated_thinking_tag_count = sum(
            decoded_response.count(tag) for tag in ("<think>", "</think>")
        )
        if not ENABLE_THINKING and generated_thinking_tag_count:
            raise RuntimeError(
                f"Non-thinking rollout emitted thinking tags at dataset index {dataset_index}"
            )
        topk_file = None
        if LOSS_SUPPORT_TOPK:
            topk_rows = rollout_topk_ids(result)
            if len(topk_rows) != len(response_ids):
                raise RuntimeError("Student top-k/response length mismatch")
            topk_file = f"sample{local_index:02d}.student_topk.pt"
            topk_path = rank_dir / topk_file
            topk_tmp = topk_path.with_suffix(topk_path.suffix + ".tmp")
            torch.save(
                torch.tensor(topk_rows, dtype=torch.int32).contiguous(), topk_tmp
            )
            topk_tmp.replace(topk_path)
        extra = rows[dataset_index]["extra_info"]
        trace = {
            "status": "ROLLOUT_OK",
            "step": args.step,
            "rank": args.rank,
            "local_index": local_index,
            "dataset_index": dataset_index,
            "source_index": int(extra["source_index"]),
            "difficulty": float(extra["difficulty"]),
            "topic": str(extra["topic"]),
            "prompt_token_ids": prompt_ids,
            "response_token_ids": response_ids,
            "prompt_tokens": len(prompt_ids),
            "response_tokens": len(response_ids),
            "requested_max_response_tokens": MAX_RESPONSE_TOKENS,
            "effective_max_response_tokens": effective_max_response_tokens,
            "thinking": ENABLE_THINKING,
            "generated_thinking_tag_count": generated_thinking_tag_count,
            "finish_reason": result.finish_reason,
            "stop_reason": getattr(result, "stop_reason", None),
            "loss_support_topk": LOSS_SUPPORT_TOPK,
            "student_topk_file": topk_file,
        }
        atomic_json(rank_dir / f"sample{local_index:02d}.json", trace)
        trace_rows.append({key: trace[key] for key in (
            "local_index", "dataset_index", "source_index", "difficulty",
            "prompt_tokens", "response_tokens", "effective_max_response_tokens",
            "generated_thinking_tag_count", "finish_reason"
        )})
    rank_manifest = {
        "status": "ROLLOUT_RANK_OK",
        "step": args.step,
        "rank": args.rank,
        "examples": len(trace_rows),
        "elapsed_seconds": time.monotonic() - started,
        "cuda": cuda_metrics(torch),
        "loss_support_topk": LOSS_SUPPORT_TOPK,
        "thinking": ENABLE_THINKING,
        "generated_thinking_tag_count": sum(
            row["generated_thinking_tag_count"] for row in trace_rows
        ),
        "rows": trace_rows,
    }
    atomic_json(rank_dir / "manifest.json", rank_manifest)
    print("FULL_VOCAB_RKL_ROLLOUT_RANK_OK", json.dumps(rank_manifest, sort_keys=True))


def phase_merge_rollout(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    for rank in range(STUDENT_WORLD_SIZE):
        manifest = read_json(args.step_dir / "traces" / f"rank{rank}" / "manifest.json")
        if manifest["status"] != "ROLLOUT_RANK_OK" or int(manifest["examples"]) <= 0:
            raise RuntimeError(f"Incomplete rollout rank {rank}")
        rows.extend(manifest["rows"])
    expected_examples = min(
        GLOBAL_BATCH_SIZE,
        TRAIN_ROWS - (args.step - 1) * GLOBAL_BATCH_SIZE,
    )
    if len(rows) != expected_examples:
        raise RuntimeError("Incomplete global rollout batch")
    merged = {
        "status": "ROLLOUT_GLOBAL_OK",
        "step": args.step,
        "examples": len(rows),
        "total_response_tokens": sum(row["response_tokens"] for row in rows),
        "max_response_tokens_observed": max(row["response_tokens"] for row in rows),
        "min_effective_max_response_tokens": min(
            row["effective_max_response_tokens"] for row in rows
        ),
        "generated_thinking_tag_count": sum(
            row["generated_thinking_tag_count"] for row in rows
        ),
        "finish_reasons": {
            reason: sum(row["finish_reason"] == reason for row in rows)
            for reason in sorted({row["finish_reason"] for row in rows})
        },
        "rows": rows,
    }
    atomic_json(args.step_dir / "rollout_manifest.json", merged)
    (args.step_dir / "ROLLOUT_READY").touch()
    print("FULL_VOCAB_RKL_ROLLOUT_GLOBAL_OK", json.dumps(merged, sort_keys=True))


def load_lm(path: Path, dtype: Any, training: bool) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.to("cuda")
    if training:
        model.train()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model


def selected_lm_logits(model: Any, hidden: Any, token_ids: Any, torch: Any) -> Any:
    """Project hidden states only onto the selected model-head rows."""
    if hidden.ndim != 3 or hidden.shape[0] != 1 or token_ids.ndim != 2:
        raise RuntimeError(f"Invalid selected-logit inputs: {hidden.shape}, {token_ids.shape}")
    positions, width = token_ids.shape
    if hidden.shape[1] != positions:
        raise RuntimeError("Selected token ids do not align with hidden positions")
    flat = token_ids.reshape(-1).long()
    weights = model.lm_head.weight.index_select(0, flat).reshape(
        positions, width, hidden.shape[-1]
    )
    logits = torch.einsum("bph,pkh->pk", hidden, weights)
    if model.lm_head.bias is not None:
        logits = logits + model.lm_head.bias.index_select(0, flat).reshape(positions, width)
    return logits


def phase_teacher(args: argparse.Namespace) -> None:
    import torch

    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    wait_for(args.step_dir / "ROLLOUT_READY")
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model = load_lm(TEACHER_MODEL, torch.bfloat16, training=False)
    completed: list[dict[str, Any]] = []
    trace_manifest = read_json(
        args.step_dir / "traces" / f"rank{args.rank}" / "manifest.json"
    )
    local_examples = int(trace_manifest["examples"])
    for local_index in range(local_examples):
        trace = read_json(
            args.step_dir / "traces" / f"rank{args.rank}" / f"sample{local_index:02d}.json"
        )
        prompt_tokens = int(trace["prompt_tokens"])
        response_tokens = int(trace["response_tokens"])
        values = trace["prompt_token_ids"] + trace["response_token_ids"]
        input_ids = torch.tensor(values, dtype=torch.long, device="cuda").unsqueeze(0)
        output_dir = args.step_dir / "teacher" / f"rank{args.rank}" / f"sample{local_index:02d}"
        output_dir.mkdir(parents=True, exist_ok=False)
        sample_started = time.monotonic()
        chunks: list[dict[str, Any]] = []
        selected_ids = None
        if LOSS_SUPPORT_TOPK:
            selected_ids = torch.load(
                args.step_dir / "traces" / f"rank{args.rank}" / trace["student_topk_file"],
                map_location="cpu", weights_only=True,
            ).long()
            if selected_ids.shape != (response_tokens, LOSS_SUPPORT_TOPK):
                raise RuntimeError(f"Teacher selected-id shape mismatch: {selected_ids.shape}")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
            position0 = prompt_tokens - 1
            for chunk_index, start in enumerate(range(0, response_tokens, CHUNK_TOKENS)):
                end = min(start + CHUNK_TOKENS, response_tokens)
                chunk_hidden = hidden[:, position0 + start : position0 + end]
                if LOSS_SUPPORT_TOPK:
                    logits = selected_lm_logits(
                        model, chunk_hidden,
                        selected_ids[start:end].to(device="cuda", non_blocking=True),
                        torch,
                    )
                    expected_width = LOSS_SUPPORT_TOPK
                else:
                    logits = model.lm_head(chunk_hidden).squeeze(0)
                    expected_width = MODEL_VOCAB_SIZE
                if logits.shape != (end - start, expected_width):
                    raise RuntimeError(f"Teacher shape mismatch: {logits.shape}")
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Non-finite teacher logits")
                cpu_logits = logits.to(dtype=torch.bfloat16, device="cpu").contiguous()
                path = output_dir / f"chunk_{chunk_index:04d}.pt"
                temporary = path.with_suffix(".pt.tmp")
                torch.save(cpu_logits, temporary)
                temporary.replace(path)
                chunks.append({
                    "index": chunk_index, "start": start, "end": end,
                    "tokens": end - start, "shape": list(cpu_logits.shape),
                    "bytes": path.stat().st_size, "file": path.name,
                })
                del chunk_hidden, logits, cpu_logits
        del hidden, input_ids, selected_ids
        sample_manifest = {
            "status": "TEACHER_RKL_SUPPORT_OK",
            "step": args.step,
            "rank": args.rank,
            "local_index": local_index,
            "dataset_index": trace["dataset_index"],
            "response_tokens": response_tokens,
            "model_vocab_size": MODEL_VOCAB_SIZE,
            "loss_support_topk": LOSS_SUPPORT_TOPK,
            "chunks": chunks,
            "elapsed_seconds": time.monotonic() - sample_started,
            "bytes": sum(chunk["bytes"] for chunk in chunks),
        }
        atomic_json(output_dir / "manifest.json", sample_manifest)
        completed.append({
            "local_index": local_index,
            "response_tokens": response_tokens,
            "elapsed_seconds": sample_manifest["elapsed_seconds"],
            "bytes": sample_manifest["bytes"],
        })
        gc.collect()
        torch.cuda.empty_cache()
    rank_manifest = {
        "status": "TEACHER_RANK_OK",
        "step": args.step,
        "rank": args.rank,
        "examples": len(completed),
        "elapsed_seconds": time.monotonic() - started,
        "cuda": cuda_metrics(torch),
        "rows": completed,
    }
    atomic_json(args.step_dir / "teacher" / f"rank{args.rank}" / "manifest.json", rank_manifest)
    print("FULL_VOCAB_RKL_TEACHER_RANK_OK", json.dumps(rank_manifest, sort_keys=True))


def parameter_checksum(model: Any, torch: Any) -> float:
    total = torch.zeros((), dtype=torch.float64, device="cuda")
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            if index >= 8:
                break
            flat = parameter.detach().reshape(-1)
            total += flat[: min(4096, flat.numel())].double().sum()
    return float(total.cpu())


def export_current_model(model: Any, run_dir: Path) -> Path:
    target = run_dir / "current_hf"
    temporary = run_dir / f".current_hf.tmp.{os.getpid()}"
    previous = run_dir / f".current_hf.previous.{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    # Nexdata accepted the first ~4.97 GB shard but returned EFBIG while
    # safetensors opened the following shard in a later export.  Smaller
    # shards avoid that storage-client boundary without changing tensors or
    # model numerics.
    model.save_pretrained(
        temporary,
        safe_serialization=True,
        max_shard_size=HF_MAX_SHARD_SIZE,
    )
    if target.exists():
        target.rename(previous)
    temporary.rename(target)
    if previous.exists():
        shutil.rmtree(previous)
    return target


def save_checkpoint(
    model: Any,
    optimizer: Any,
    checkpoint_root: Path,
    run_dir: Path,
    step: int,
    metrics: dict[str, Any],
    torch: Any,
) -> None:
    final_dir = checkpoint_root / f"global_step_{step}"
    if final_dir.exists():
        raise FileExistsError(final_dir)
    temporary = checkpoint_root / f".global_step_{step}.tmp.{os.getpid()}"
    actor = temporary / "actor"
    actor.mkdir(parents=True, exist_ok=False)
    shutil.copytree(run_dir / "current_hf", actor / "huggingface")
    torch.save(model.state_dict(), actor / "model.pt")
    torch.save(optimizer.state_dict(), actor / "optimizer.pt")
    torch.save(
        {
            "step": step,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(),
            "metrics": metrics,
        },
        actor / "extra_state.pt",
    )
    atomic_json(temporary / "checkpoint_manifest.json", {
        "status": "DEEPMATH_RKL_CHECKPOINT_OK",
        "step": step,
        "objective": OBJECTIVE,
        "model_vocab_size": MODEL_VOCAB_SIZE,
        "loss_support_topk": LOSS_SUPPORT_TOPK,
        "metrics": metrics,
    })
    temporary.rename(final_dir)
    latest_tmp = checkpoint_root / "latest_checkpointed_iteration.txt.tmp"
    latest_tmp.write_text(f"{step}\n", encoding="utf-8")
    latest_tmp.replace(checkpoint_root / "latest_checkpointed_iteration.txt")


def latest_checkpoint(checkpoint_root: Path) -> tuple[int, Path]:
    latest = checkpoint_root / "latest_checkpointed_iteration.txt"
    if not latest.exists():
        return 0, STUDENT_MODEL
    step = int(latest.read_text(encoding="utf-8").strip())
    if step not in SAVE_STEPS:
        raise RuntimeError(f"Unexpected checkpoint step {step}")
    path = checkpoint_root / f"global_step_{step}"
    if not (path / "checkpoint_manifest.json").exists():
        raise RuntimeError(f"Incomplete checkpoint {path}")
    return step, path / "actor" / "huggingface"


def phase_student_daemon(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F

    # Variable-length reasoning traces can leave faster ranks waiting well over
    # NCCL's default timeout before the global gradient reduction.
    dist.init_process_group("nccl", timeout=timedelta(hours=6))
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world = dist.get_world_size()
    if world != STUDENT_WORLD_SIZE:
        raise RuntimeError(f"Expected {STUDENT_WORLD_SIZE} student ranks, got {world}")
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    checkpoint_step, model_path = latest_checkpoint(args.checkpoint_root)
    model = load_lm(model_path, torch.float32, training=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-6, weight_decay=0.01, betas=(0.9, 0.999)
    )
    if checkpoint_step:
        state_path = args.checkpoint_root / f"global_step_{checkpoint_step}" / "actor"
        optimizer.load_state_dict(torch.load(
            state_path / "optimizer.pt", map_location="cuda", weights_only=False
        ))
        extra = torch.load(state_path / "extra_state.pt", map_location="cpu", weights_only=False)
        torch.set_rng_state(extra["torch_rng_state"])
        torch.cuda.set_rng_state(extra["cuda_rng_state"].to("cpu"))
    if rank == 0:
        args.checkpoint_root.mkdir(parents=True, exist_ok=True)
        export_current_model(model, args.run_dir)
        atomic_json(args.run_dir / "student_daemon_ready.json", {
            "status": "STUDENT_DAEMON_READY",
            "checkpoint_step": checkpoint_step,
            "world_size": world,
            "model_path": str(model_path),
        })
    dist.barrier()

    for step in range(checkpoint_step + 1, TOTAL_STEPS + 1):
        step_dir = args.run_dir / "steps" / f"step_{step:04d}"
        wait_for(step_dir / "ROLLOUT_READY", timeout_seconds=172800)
        rollout = read_json(step_dir / "rollout_manifest.json")
        global_tokens = int(rollout["total_response_tokens"])
        if rollout["examples"] != GLOBAL_BATCH_SIZE or global_tokens <= 0:
            raise RuntimeError("Invalid rollout manifest")
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        checksum_before = parameter_checksum(model, torch)
        local_loss_sum = 0.0
        local_tokens = 0
        sample_rows: list[dict[str, Any]] = []
        trace_manifest = read_json(
            step_dir / "traces" / f"rank{rank}" / "manifest.json"
        )
        local_examples = int(trace_manifest["examples"])
        for local_index in range(local_examples):
            trace_path = step_dir / "traces" / f"rank{rank}" / f"sample{local_index:02d}.json"
            trace = read_json(trace_path)
            teacher_dir = step_dir / "teacher" / f"rank{rank}" / f"sample{local_index:02d}"
            wait_for(teacher_dir / "manifest.json", timeout_seconds=172800)
            teacher = read_json(teacher_dir / "manifest.json")
            response_tokens = int(trace["response_tokens"])
            if teacher["response_tokens"] != response_tokens:
                raise RuntimeError("Teacher/student token mismatch")
            selected_ids = None
            if LOSS_SUPPORT_TOPK:
                selected_ids = torch.load(
                    step_dir / "traces" / f"rank{rank}" / trace["student_topk_file"],
                    map_location="cpu", weights_only=True,
                ).long()
                if selected_ids.shape != (response_tokens, LOSS_SUPPORT_TOPK):
                    raise RuntimeError(f"Student selected-id shape mismatch: {selected_ids.shape}")
            ids = trace["prompt_token_ids"] + trace["response_token_ids"]
            input_ids = torch.tensor(ids, dtype=torch.long, device="cuda").unsqueeze(0)
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
                    if LOSS_SUPPORT_TOPK:
                        student_logits = selected_lm_logits(
                            model, chunk_hidden,
                            selected_ids[start:end].to(device="cuda", non_blocking=True),
                            torch,
                        )
                    else:
                        student_logits = model.lm_head(chunk_hidden).squeeze(0)
                if student_logits.shape != teacher_logits.shape:
                    raise RuntimeError("Teacher/student support shape mismatch")
                student_logp = F.log_softmax(student_logits.float(), dim=-1)
                teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
                probabilities = student_logp.exp()
                token_kl = (probabilities * (student_logp - teacher_logp)).sum(dim=-1)
                if not torch.isfinite(token_kl).all():
                    raise FloatingPointError("Non-finite Reverse KL")
                chunk_sum = token_kl.sum()
                (chunk_sum / global_tokens).backward()
                hidden_gradient[:, position0 + start : position0 + end].copy_(chunk_hidden.grad)
                value = float(chunk_sum.detach().cpu())
                sample_loss += value
                del teacher_logits, chunk_hidden, student_logits, student_logp
                del teacher_logp, probabilities, token_kl, chunk_sum
            hidden.backward(hidden_gradient)
            local_loss_sum += sample_loss
            local_tokens += response_tokens
            sample_rows.append({
                "local_index": local_index,
                "dataset_index": trace["dataset_index"],
                "response_tokens": response_tokens,
                "mean_reverse_kl": sample_loss / response_tokens,
            })
            del input_ids, hidden, hidden_gradient, selected_ids
            shutil.rmtree(teacher_dir)
            gc.collect()
            torch.cuda.empty_cache()

        # Local losses were normalized by global token count; summing gradients
        # across ranks therefore yields the exact global token-mean gradient.
        for parameter in model.parameters():
            if parameter.grad is None:
                raise RuntimeError("A trainable parameter is missing its gradient")
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_norm = float(grad_norm_tensor.detach().cpu())
        if not math.isfinite(grad_norm):
            raise FloatingPointError("Non-finite gradient norm")
        optimizer.step()
        checksum_after = parameter_checksum(model, torch)
        local_metrics = {
            "rank": rank,
            "local_tokens": local_tokens,
            "local_loss_sum": local_loss_sum,
            "elapsed_seconds": time.monotonic() - started,
            "cuda": cuda_metrics(torch),
            "samples": sample_rows,
        }
        gathered: list[Any] | None = [None] * world if rank == 0 else None
        dist.gather_object(local_metrics, gathered, dst=0)
        if rank == 0:
            assert gathered is not None
            metrics = {
                "status": "DEEPMATH_RKL_STEP_OK",
                "step": step,
                "objective": OBJECTIVE,
                "loss_support_topk": LOSS_SUPPORT_TOPK,
                "global_batch_size": GLOBAL_BATCH_SIZE,
                "global_response_tokens": global_tokens,
                "mean_reverse_kl": sum(row["local_loss_sum"] for row in gathered) / global_tokens,
                "gradient_norm_before_clip": grad_norm,
                "parameter_checksum_before": checksum_before,
                "parameter_checksum_after": checksum_after,
                "parameter_checksum_delta": checksum_after - checksum_before,
                "rank_metrics": gathered,
            }
            if metrics["parameter_checksum_delta"] == 0.0:
                raise RuntimeError("Optimizer step did not change parameter checksum")
            atomic_json(step_dir / "step_metrics.json", metrics)
            export_current_model(model, args.run_dir)
            if step in SAVE_STEPS:
                save_checkpoint(model, optimizer, args.checkpoint_root, args.run_dir, step, metrics, torch)
            (step_dir / "STUDENT_DONE").touch()
            print("DEEPMATH_RKL_STEP_OK", json.dumps(metrics, sort_keys=True), flush=True)
        # Gradients are not part of optimizer state. Releasing them leaves
        # ample room for the co-resident vLLM rollout engine on the next step.
        # Rank 0 additionally serializes the HF snapshot above; safetensors can
        # leave its temporary CUDA allocator blocks reserved after returning.
        # Release all unreferenced blocks on every rank before the barrier so
        # the next vLLM engine sees the same clean memory headroom.
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
    if rank == 0:
        (args.run_dir / "TRAINING_DONE").touch()
    dist.barrier()
    dist.destroy_process_group()


def phase_step_summary(args: argparse.Namespace) -> None:
    wait_for(args.step_dir / "STUDENT_DONE", timeout_seconds=172800)
    student = read_json(args.step_dir / "step_metrics.json")
    rollout = read_json(args.step_dir / "rollout_manifest.json")
    teachers = [
        read_json(args.step_dir / "teacher" / f"rank{rank}" / "manifest.json")
        for rank in range(STUDENT_WORLD_SIZE)
    ]
    if any(item["status"] != "TEACHER_RANK_OK" for item in teachers):
        raise RuntimeError("Teacher rank incomplete")
    summary = {
        "status": "DEEPMATH_RKL_STEP_COMPLETE",
        "step": args.step,
        "rollout": rollout,
        "student": student,
        "teacher_elapsed_max_seconds": max(item["elapsed_seconds"] for item in teachers),
        "teacher_peak_allocated_max_gib": max(item["cuda"]["peak_allocated_gib"] for item in teachers),
    }
    metrics_dir = args.run_dir / "metrics"
    atomic_json(metrics_dir / f"step_{args.step:04d}.json", summary)
    # Preserve compact evidence only. Full token traces and logits are transient.
    shutil.rmtree(args.step_dir)
    print("DEEPMATH_RKL_STEP_COMPLETE", json.dumps(summary, sort_keys=True))


def phase_verify(args: argparse.Namespace) -> None:
    latest = args.checkpoint_root / "latest_checkpointed_iteration.txt"
    if not latest.exists() or int(latest.read_text().strip()) != TOTAL_STEPS:
        raise RuntimeError(f"Final checkpoint pointer is not step {TOTAL_STEPS}")
    found = sorted(
        int(path.name.removeprefix("global_step_"))
        for path in args.checkpoint_root.glob("global_step_*")
        if path.is_dir()
    )
    if found != list(SAVE_STEPS):
        raise RuntimeError(f"Checkpoint set mismatch: {found}")
    for step in SAVE_STEPS:
        root = args.checkpoint_root / f"global_step_{step}"
        for relative in (
            "checkpoint_manifest.json", "actor/model.pt", "actor/optimizer.pt",
            "actor/extra_state.pt", "actor/huggingface/config.json",
        ):
            if not (root / relative).exists():
                raise FileNotFoundError(root / relative)
    result = {
        "status": "DEEPMATH_RKL_COMPLETE",
        "objective": OBJECTIVE,
        "loss_support_topk": LOSS_SUPPORT_TOPK,
        "total_steps": TOTAL_STEPS,
        "checkpoints": list(SAVE_STEPS),
        "thinking": ENABLE_THINKING,
    }
    atomic_json(args.run_dir / "final_result.json", result)
    print("DEEPMATH_RKL_COMPLETE", json.dumps(result, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=(
        "init", "rollout", "merge-rollout", "teacher", "student-daemon",
        "step-summary", "verify",
    ))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    args.step_dir = args.run_dir / "steps" / f"step_{args.step:04d}"
    if args.phase in {"student-daemon", "verify"} and args.checkpoint_root is None:
        parser.error("--checkpoint-root is required")
    if args.phase == "rollout" and args.model_dir is None:
        parser.error("--model-dir is required")
    return args


def main() -> None:
    args = parse_args()
    phase = {
        "init": phase_init,
        "rollout": phase_rollout,
        "merge-rollout": phase_merge_rollout,
        "teacher": phase_teacher,
        "student-daemon": phase_student_daemon,
        "step-summary": phase_step_summary,
        "verify": phase_verify,
    }[args.phase]
    phase(args)


if __name__ == "__main__":
    main()
