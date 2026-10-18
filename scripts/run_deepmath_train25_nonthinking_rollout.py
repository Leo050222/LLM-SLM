#!/usr/bin/env python3
"""Generate lossless native non-thinking Qwen3 rollout traces for one shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def user_prompt(problem: str) -> str:
    return (
        "Solve the following math problem. Show your reasoning and put your "
        "final answer inside \\boxed{}.\n\n" + problem
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--expected-rows", type=int, default=888)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--max-tokens", type=int, default=40264)
    parser.add_argument("--max-model-len", type=int, default=40960)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    observed_input_sha = sha256(args.input)
    if observed_input_sha != args.input_sha256:
        raise RuntimeError(f"Input SHA mismatch: {observed_input_sha}")
    frame = pd.read_parquet(args.input)
    if len(frame) != args.expected_rows:
        raise RuntimeError(f"Expected {args.expected_rows} rows, got {len(frame)}")
    if any(int(value) % args.num_shards != args.shard_index for value in frame.dataset_index):
        raise RuntimeError("Modulo-shard membership mismatch")
    if args.limit is not None:
        if args.limit <= 0 or args.limit > len(frame):
            raise ValueError(f"Invalid limit: {args.limit}")
        frame = frame.iloc[: args.limit]

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    prompts: list[TokensPrompt] = []
    metadata: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        message = user_prompt(str(row.problem))
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": message}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
        if not prompt_ids or not all(isinstance(token_id, int) for token_id in prompt_ids):
            raise TypeError("Chat template did not return integer prompt token IDs")
        prompts.append(TokensPrompt(prompt_token_ids=prompt_ids))
        metadata.append(
            {
                "dataset": "deepmath_ge6_train25pct_perdifficulty",
                "dataset_index": int(row.dataset_index),
                "problem_id": str(row.problem_id),
                "problem": str(row.problem),
                "solution": str(row.solution),
                "difficulty": float(row.difficulty),
                "topic": str(row.topic),
                "source_index": int(row.source_index),
                "padded_duplicate": bool(row.padded_duplicate),
                "prompt_text": message,
                "prompt_token_ids": list(prompt_ids),
            }
        )

    engine = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=False,
        seed=args.seed,
        gpu_memory_utilization=0.90,
        max_model_len=args.max_model_len,
        max_num_seqs=8,
        enable_prefix_caching=True,
        disable_log_stats=True,
    )
    sampling = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=args.max_tokens,
        n=1,
        seed=args.seed,
    )
    started = time.time()
    outputs = engine.generate(prompts, sampling, use_tqdm=True)
    elapsed = time.time() - started
    rows: list[dict[str, Any]] = []
    for item, request_output in zip(metadata, outputs, strict=True):
        generated = request_output.outputs[0]
        response_ids = [int(value) for value in generated.token_ids]
        response = tokenizer.decode(response_ids, skip_special_tokens=False)
        if "<think>" in response or "</think>" in response:
            raise RuntimeError(
                f"Generated thinking tag at dataset_index={item['dataset_index']}"
            )
        rows.append(
            {
                **item,
                "response": response,
                "response_token_ids": response_ids,
                "response_tokens": len(response_ids),
                "finish_reason": generated.finish_reason,
                "stop_reason": generated.stop_reason,
            }
        )

    args.output_dir.mkdir(parents=True)
    rollout_path = args.output_dir / "rollouts.jsonl"
    atomic_jsonl(rollout_path, rows)
    config = {
        "status": "DEEPMATH_TRAIN25_NATIVE_NONTHINKING_SHARD_OK",
        "input": str(args.input),
        "input_sha256": observed_input_sha,
        "output_sha256": sha256(rollout_path),
        "model": args.model,
        "tokenizer": args.tokenizer,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "num_prompts": len(rows),
        "limit": args.limit,
        "padding_rows": sum(row["padded_duplicate"] for row in rows),
        "thinking_mode": False,
        "parameters_frozen": True,
        "optimizer_created": False,
        "backward_called": False,
        "sampling": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "n": 1,
            "seed": args.seed,
        },
        "elapsed_seconds_generate": elapsed,
        "total_response_tokens": sum(row["response_tokens"] for row in rows),
        "vllm_port": os.environ.get("VLLM_PORT"),
        "slurm_job_id": os.environ.get("SLURM_ARRAY_JOB_ID")
        or os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "worker_rank": os.environ.get("OPD_WORKER_RANK"),
        "node": os.environ.get("SLURMD_NODENAME"),
    }
    atomic_json(args.output_dir / "run_config.json", config)
    (args.output_dir / "SHARD_COMPLETE").touch()
    print(json.dumps(config, ensure_ascii=False, sort_keys=True))
    print("DEEPMATH_TRAIN25_NATIVE_NONTHINKING_SHARD_OK")


if __name__ == "__main__":
    main()
