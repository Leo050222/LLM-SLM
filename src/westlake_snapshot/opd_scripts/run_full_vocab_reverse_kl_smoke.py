#!/usr/bin/env python3
"""One-step exact full-vocabulary Reverse-KL smoke for Qwen3-1.7B <- Qwen3-14B.

The smoke deliberately separates rollout, frozen-teacher scoring, and student
backpropagation. Teacher logits are materialized in bounded token-position
chunks; no vocabulary truncation or sampled-token KL estimator is used.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


STUDENT_MODEL = Path("/storage/yukaichengLab/share/model/Qwen3-1.7B")
TEACHER_MODEL = Path("/storage/yukaichengLab/share/model/Qwen3-14B")
SMOKE_DATA = Path(
    "/storage/yukaichengLab/wangxin/test_eiloeo/opd_data/"
    "deepmath_ge6_opd/smoke_train.parquet"
)
SEED = 20260810
MAX_MODEL_LEN = 40960
MAX_RESPONSE_TOKENS = 40264
# Qwen3's tokenizer has 151,669 entries, while both local causal-LM heads expose
# 151,936 logits. Exact KL must include every model-head logit, including the
# 267 reserved/padded slots, rather than silently trimming to tokenizer length.
TOKENIZER_VOCAB_SIZE = 151669
MODEL_VOCAB_SIZE = 151936
DEFAULT_CHUNK_TOKENS = 256


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


def cuda_memory_gib(torch_module: Any) -> dict[str, float]:
    torch = torch_module
    return {
        "allocated_gib": torch.cuda.memory_allocated() / (1024**3),
        "reserved_gib": torch.cuda.memory_reserved() / (1024**3),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
    }


def load_smoke_row(rank: int) -> dict[str, Any]:
    import pyarrow.parquet as pq

    rows = pq.read_table(SMOKE_DATA).to_pylist()
    if len(rows) != 4:
        raise RuntimeError(f"Expected four smoke rows, found {len(rows)}")
    if not 0 <= rank < len(rows):
        raise IndexError(rank)
    return rows[rank]


def phase_generate(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    torch.cuda.set_device(0)
    row = load_smoke_row(args.rank)
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, local_files_only=True)
    messages = row["prompt"]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_dict=True,
    )
    prompt_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if prompt_ids and isinstance(prompt_ids[0], list):
        if len(prompt_ids) != 1:
            raise RuntimeError("Unexpected batched prompt")
        prompt_ids = prompt_ids[0]
    prompt_ids = [int(value) for value in prompt_ids]
    if len(prompt_ids) > 695:
        raise RuntimeError(f"Prompt exceeds configured limit: {len(prompt_ids)}")

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    llm = LLM(
        model=str(STUDENT_MODEL),
        tokenizer=str(STUDENT_MODEL),
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=False,
        seed=SEED,
        gpu_memory_utilization=0.90,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=1,
        enable_prefix_caching=False,
        enforce_eager=True,
        disable_log_stats=True,
    )
    sampling = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=MAX_RESPONSE_TOKENS,
        n=1,
        seed=SEED,
    )
    result = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids)], sampling, use_tqdm=True
    )[0].outputs[0]
    response_ids = [int(value) for value in result.token_ids]
    if not response_ids:
        raise RuntimeError("Student generated an empty response")
    if len(prompt_ids) + len(response_ids) + 1 > MAX_MODEL_LEN:
        raise RuntimeError("Trace exceeds teacher next-token scoring context")
    output = {
        "status": "ROLLOUT_OK",
        "rank": args.rank,
        "seed": SEED,
        "student_model": str(STUDENT_MODEL),
        "prompt_token_ids": prompt_ids,
        "response_token_ids": response_ids,
        "prompt_tokens": len(prompt_ids),
        "response_tokens": len(response_ids),
        "finish_reason": result.finish_reason,
        "stop_reason": getattr(result, "stop_reason", None),
        "elapsed_seconds": time.monotonic() - started,
        "difficulty": float(row["extra_info"]["difficulty"]),
        "topic": str(row["extra_info"]["topic"]),
        "source_index": int(row["extra_info"]["source_index"]),
        "cuda": cuda_memory_gib(torch),
    }
    atomic_json(args.output_dir / f"trace_rank{args.rank}.json", output)
    print("FULL_VOCAB_RKL_ROLLOUT_OK", json.dumps(output, sort_keys=True))


def load_causal_lm(model_path: Path, dtype: Any, training: bool) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
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


def trace_input_ids(trace: dict[str, Any], torch_module: Any) -> Any:
    torch = torch_module
    values = trace["prompt_token_ids"] + trace["response_token_ids"]
    return torch.tensor(values, dtype=torch.long, device="cuda").unsqueeze(0)


def phase_teacher(args: argparse.Namespace) -> None:
    import torch

    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    trace = read_json(args.output_dir / f"trace_rank{args.rank}.json")
    if trace["status"] != "ROLLOUT_OK":
        raise RuntimeError("Rollout is incomplete")
    response_tokens = int(trace["response_tokens"])
    prompt_tokens = int(trace["prompt_tokens"])
    rank_dir = args.output_dir / "teacher_logits" / f"rank{args.rank}"
    rank_dir.mkdir(parents=True, exist_ok=False)

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model = load_causal_lm(TEACHER_MODEL, torch.bfloat16, training=False)
    input_ids = trace_input_ids(trace, torch)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        start_position = prompt_tokens - 1
        chunks: list[dict[str, Any]] = []
        for chunk_index, start in enumerate(
            range(0, response_tokens, args.chunk_tokens)
        ):
            end = min(start + args.chunk_tokens, response_tokens)
            positions = hidden[:, start_position + start : start_position + end]
            logits = model.lm_head(positions).squeeze(0)
            if logits.shape != (end - start, MODEL_VOCAB_SIZE):
                raise RuntimeError(f"Unexpected teacher logits shape: {logits.shape}")
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite teacher logits")
            cpu_logits = logits.to(dtype=torch.bfloat16, device="cpu").contiguous()
            chunk_path = rank_dir / f"chunk_{chunk_index:04d}.pt"
            temporary = chunk_path.with_suffix(".pt.tmp")
            torch.save(cpu_logits, temporary)
            temporary.replace(chunk_path)
            chunks.append(
                {
                    "index": chunk_index,
                    "start": start,
                    "end": end,
                    "tokens": end - start,
                    "shape": list(cpu_logits.shape),
                    "bytes": chunk_path.stat().st_size,
                    "file": chunk_path.name,
                }
            )
            del positions, logits, cpu_logits

    manifest = {
        "status": "TEACHER_FULL_VOCAB_OK",
        "rank": args.rank,
        "teacher_model": str(TEACHER_MODEL),
        "dtype": "bfloat16",
        "model_vocab_size": MODEL_VOCAB_SIZE,
        "tokenizer_vocab_size": TOKENIZER_VOCAB_SIZE,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "chunk_tokens": args.chunk_tokens,
        "chunks": chunks,
        "total_chunk_bytes": sum(item["bytes"] for item in chunks),
        "elapsed_seconds": time.monotonic() - started,
        "cuda": cuda_memory_gib(torch),
    }
    atomic_json(rank_dir / "manifest.json", manifest)
    print("FULL_VOCAB_RKL_TEACHER_OK", json.dumps(manifest, sort_keys=True))


def parameter_checksum(model: Any, torch_module: Any) -> float:
    torch = torch_module
    total = torch.zeros((), dtype=torch.float64, device="cuda")
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            if index >= 8:
                break
            flat = parameter.detach().reshape(-1)
            total += flat[: min(flat.numel(), 4096)].double().sum()
    return float(total.cpu())


def wait_for_manifest(path: Path, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.exists():
            return read_json(path)
        if (path.parent / "FAILED").exists():
            raise RuntimeError(f"Teacher process failed before writing {path}")
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {path}")


def phase_student(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F

    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    trace = read_json(args.output_dir / f"trace_rank{args.rank}.json")
    response_tokens = int(trace["response_tokens"])
    prompt_tokens = int(trace["prompt_tokens"])
    teacher_dir = args.output_dir / "teacher_logits" / f"rank{args.rank}"

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    # FP32 master parameters make the optimizer-step smoke meaningful at lr=1e-6.
    model = load_causal_lm(STUDENT_MODEL, torch.float32, training=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-6, weight_decay=0.01, betas=(0.9, 0.999)
    )
    checksum_before = parameter_checksum(model, torch)

    teacher_manifest = wait_for_manifest(
        teacher_dir / "manifest.json", timeout_seconds=args.teacher_wait_seconds
    )
    if teacher_manifest["status"] != "TEACHER_FULL_VOCAB_OK":
        raise RuntimeError("Teacher manifest is incomplete")
    if int(teacher_manifest["response_tokens"]) != response_tokens:
        raise RuntimeError("Teacher/student response token count mismatch")
    if int(teacher_manifest["model_vocab_size"]) != MODEL_VOCAB_SIZE:
        raise RuntimeError("Teacher vocabulary mismatch")

    input_ids = trace_input_ids(trace, torch)
    attention_mask = torch.ones_like(input_ids)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
    if not hidden.requires_grad:
        raise RuntimeError("Student hidden states do not require gradients")

    start_position = prompt_tokens - 1
    hidden_gradient = torch.zeros_like(hidden)
    loss_sum = 0.0
    loss_min = math.inf
    loss_max = -math.inf
    processed_tokens = 0
    chunk_metrics: list[dict[str, Any]] = []

    for chunk in teacher_manifest["chunks"]:
        start = int(chunk["start"])
        end = int(chunk["end"])
        teacher_logits = torch.load(
            teacher_dir / str(chunk["file"]), map_location="cpu", weights_only=True
        ).to(device="cuda", dtype=torch.bfloat16)
        chunk_hidden = (
            hidden[:, start_position + start : start_position + end]
            .detach()
            .requires_grad_(True)
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            student_logits_bf16 = model.lm_head(chunk_hidden).squeeze(0)
        if student_logits_bf16.shape != teacher_logits.shape:
            raise RuntimeError(
                f"Student/teacher shape mismatch: {student_logits_bf16.shape} "
                f"vs {teacher_logits.shape}"
            )
        student_logp = F.log_softmax(student_logits_bf16.float(), dim=-1)
        teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
        student_probability = student_logp.exp()
        token_reverse_kl = (
            student_probability * (student_logp - teacher_logp)
        ).sum(dim=-1)
        if not torch.isfinite(token_reverse_kl).all():
            raise FloatingPointError("Non-finite full-vocabulary Reverse KL")
        chunk_loss_sum = token_reverse_kl.sum()
        scaled_loss = chunk_loss_sum / response_tokens
        scaled_loss.backward()
        if chunk_hidden.grad is None:
            raise RuntimeError("Missing student hidden gradient")
        hidden_gradient[:, start_position + start : start_position + end].copy_(
            chunk_hidden.grad
        )
        values = token_reverse_kl.detach()
        chunk_loss_value = float(chunk_loss_sum.detach().cpu())
        loss_sum += chunk_loss_value
        loss_min = min(loss_min, float(values.min().cpu()))
        loss_max = max(loss_max, float(values.max().cpu()))
        processed_tokens += end - start
        chunk_metrics.append(
            {
                "index": int(chunk["index"]),
                "start": start,
                "end": end,
                "mean_reverse_kl": float(values.mean().cpu()),
                "max_reverse_kl": float(values.max().cpu()),
            }
        )
        del (
            teacher_logits,
            chunk_hidden,
            student_logits_bf16,
            student_logp,
            teacher_logp,
            student_probability,
            token_reverse_kl,
            chunk_loss_sum,
            scaled_loss,
            values,
        )
        # Periodic release keeps allocator fragmentation bounded without paying
        # the synchronization cost on every token-position chunk.
        if (int(chunk["index"]) + 1) % 16 == 0:
            torch.cuda.empty_cache()

    if processed_tokens != response_tokens:
        raise RuntimeError(
            f"Processed {processed_tokens} tokens, expected {response_tokens}"
        )
    # Propagate the accumulated exact LM-head gradient through the transformer once.
    hidden.backward(hidden_gradient)
    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    grad_norm = float(grad_norm_tensor.detach().cpu())
    if not math.isfinite(grad_norm):
        raise FloatingPointError(f"Non-finite gradient norm: {grad_norm}")
    optimizer.step()
    checksum_after = parameter_checksum(model, torch)

    metrics = {
        "status": "STUDENT_FULL_VOCAB_BACKWARD_OK",
        "objective": "exact_full_vocabulary_reverse_kl_student_to_teacher",
        "rank": args.rank,
        "student_model": str(STUDENT_MODEL),
        "teacher_model": str(TEACHER_MODEL),
        "model_vocab_size": MODEL_VOCAB_SIZE,
        "tokenizer_vocab_size": TOKENIZER_VOCAB_SIZE,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "processed_tokens": processed_tokens,
        "chunk_tokens": args.chunk_tokens,
        "chunks": len(chunk_metrics),
        "mean_reverse_kl": loss_sum / response_tokens,
        "min_token_reverse_kl": loss_min,
        "max_token_reverse_kl": loss_max,
        "gradient_norm_before_clip": grad_norm,
        "parameter_checksum_before": checksum_before,
        "parameter_checksum_after": checksum_after,
        "parameter_checksum_delta": checksum_after - checksum_before,
        "elapsed_seconds": time.monotonic() - started,
        "cuda": cuda_memory_gib(torch),
        "chunk_metrics": chunk_metrics,
    }
    if metrics["mean_reverse_kl"] < -1e-5:
        raise FloatingPointError(f"KL mean is negative: {metrics['mean_reverse_kl']}")
    atomic_json(args.output_dir / f"student_rank{args.rank}.json", metrics)
    print("FULL_VOCAB_RKL_STUDENT_OK", json.dumps(metrics, sort_keys=True))


def phase_unit(args: argparse.Namespace) -> None:
    """Prove chunked hidden-gradient accumulation matches monolithic exact KL."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(SEED)
    dtype = torch.float64
    inputs = torch.randn(7, 5, dtype=dtype)
    teacher_logits = torch.randn(7, 31, dtype=dtype)
    base_a = torch.nn.Linear(5, 13, bias=True, dtype=dtype)
    head_a = torch.nn.Linear(13, 31, bias=False, dtype=dtype)
    base_b = torch.nn.Linear(5, 13, bias=True, dtype=dtype)
    head_b = torch.nn.Linear(13, 31, bias=False, dtype=dtype)
    base_b.load_state_dict(base_a.state_dict())
    head_b.load_state_dict(head_a.state_dict())

    hidden_a = base_a(inputs)
    student_logp_a = F.log_softmax(head_a(hidden_a), dim=-1)
    teacher_logp = F.log_softmax(teacher_logits, dim=-1)
    loss_a = (
        student_logp_a.exp() * (student_logp_a - teacher_logp)
    ).sum(dim=-1).mean()
    loss_a.backward()

    hidden_b = base_b(inputs)
    hidden_grad = torch.zeros_like(hidden_b)
    loss_b_value = 0.0
    for start in range(0, 7, 3):
        end = min(start + 3, 7)
        chunk_hidden = hidden_b[start:end].detach().requires_grad_(True)
        student_logp = F.log_softmax(head_b(chunk_hidden), dim=-1)
        chunk_loss = (
            student_logp.exp() * (student_logp - teacher_logp[start:end])
        ).sum() / 7
        loss_b_value += float(chunk_loss.detach())
        chunk_loss.backward()
        hidden_grad[start:end].copy_(chunk_hidden.grad)
    hidden_b.backward(hidden_grad)

    checks = {
        "loss_abs_error": abs(float(loss_a.detach()) - loss_b_value),
        "base_weight_grad_max_error": float(
            (base_a.weight.grad - base_b.weight.grad).abs().max()
        ),
        "base_bias_grad_max_error": float(
            (base_a.bias.grad - base_b.bias.grad).abs().max()
        ),
        "head_grad_max_error": float(
            (head_a.weight.grad - head_b.weight.grad).abs().max()
        ),
    }
    if max(checks.values()) > 1e-10:
        raise AssertionError(checks)
    print("FULL_VOCAB_RKL_UNIT_OK", json.dumps(checks, sort_keys=True))


def phase_summarize(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    total_teacher_bytes = 0
    for rank in range(4):
        trace = read_json(args.output_dir / f"trace_rank{rank}.json")
        teacher = read_json(
            args.output_dir / "teacher_logits" / f"rank{rank}" / "manifest.json"
        )
        student = read_json(args.output_dir / f"student_rank{rank}.json")
        if trace["status"] != "ROLLOUT_OK":
            raise RuntimeError(f"Rank {rank} rollout incomplete")
        if teacher["status"] != "TEACHER_FULL_VOCAB_OK":
            raise RuntimeError(f"Rank {rank} teacher incomplete")
        if student["status"] != "STUDENT_FULL_VOCAB_BACKWARD_OK":
            raise RuntimeError(f"Rank {rank} student incomplete")
        if not (
            trace["response_tokens"]
            == teacher["response_tokens"]
            == student["processed_tokens"]
        ):
            raise RuntimeError(f"Rank {rank} token counts do not align")
        total_teacher_bytes += int(teacher["total_chunk_bytes"])
        rows.append(
            {
                "rank": rank,
                "source_index": trace["source_index"],
                "difficulty": trace["difficulty"],
                "prompt_tokens": trace["prompt_tokens"],
                "response_tokens": trace["response_tokens"],
                "finish_reason": trace["finish_reason"],
                "rollout_seconds": trace["elapsed_seconds"],
                "teacher_seconds": teacher["elapsed_seconds"],
                "student_seconds": student["elapsed_seconds"],
                "teacher_logit_bytes": teacher["total_chunk_bytes"],
                "mean_reverse_kl": student["mean_reverse_kl"],
                "gradient_norm": student["gradient_norm_before_clip"],
                "teacher_peak_allocated_gib": teacher["cuda"][
                    "peak_allocated_gib"
                ],
                "student_peak_allocated_gib": student["cuda"][
                    "peak_allocated_gib"
                ],
            }
        )
    total_tokens = sum(row["response_tokens"] for row in rows)
    summary = {
        "status": "DEEPMATH_FULL_VOCAB_RKL_MAXLEN_SMOKE_OK",
        "objective": "exact_full_vocabulary_reverse_kl_student_to_teacher",
        "student_model": str(STUDENT_MODEL),
        "teacher_model": str(TEACHER_MODEL),
        "model_vocab_size": MODEL_VOCAB_SIZE,
        "tokenizer_vocab_size": TOKENIZER_VOCAB_SIZE,
        "max_model_len": MAX_MODEL_LEN,
        "max_response_tokens": MAX_RESPONSE_TOKENS,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "thinking": True,
        "chunk_tokens": args.chunk_tokens,
        "examples": len(rows),
        "total_response_tokens": total_tokens,
        "total_teacher_logit_bytes": total_teacher_bytes,
        "mean_reverse_kl_token_weighted": sum(
            row["mean_reverse_kl"] * row["response_tokens"] for row in rows
        )
        / total_tokens,
        "rows": rows,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print("DEEPMATH_FULL_VOCAB_RKL_MAXLEN_SMOKE_OK", json.dumps(summary, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=["unit", "generate", "teacher", "student", "summarize"]
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument("--teacher-wait-seconds", type=float, default=21600)
    args = parser.parse_args()
    if args.chunk_tokens <= 0:
        parser.error("--chunk-tokens must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phase = {
        "unit": phase_unit,
        "generate": phase_generate,
        "teacher": phase_teacher,
        "student": phase_student,
        "summarize": phase_summarize,
    }[args.phase]
    phase(args)


if __name__ == "__main__":
    main()
