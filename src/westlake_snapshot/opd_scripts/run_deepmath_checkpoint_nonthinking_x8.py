#!/usr/bin/env python3
"""Generate one modulo shard of DeepMath non-thinking x8 rollouts.

Generation can be separated from scoring so model output is always durably
recorded, including rare protocol-violation strings that must be audited later.
"""

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

from run_opd_logit_diagnostic import (
    deepmath_user_prompt,
    score_math_response,
    write_json,
    write_jsonl,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--expected-total-rows", type=int, default=1000)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--max-tokens", type=int, default=40264)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--score-timeout", type=float, default=20.0)
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index")
    if args.num_rollouts != 8:
        raise ValueError("This strict experiment requires exactly eight rollouts")

    frame = pd.read_parquet(args.input)
    if len(frame) != args.expected_total_rows:
        raise RuntimeError(f"Expected {args.expected_total_rows} rows, got {len(frame)}")
    required = {
        "dataset_index", "id", "problem", "solution", "difficulty", "topic", "source_index"
    }
    if not required.issubset(frame.columns):
        raise RuntimeError(f"Missing columns: {sorted(required - set(frame.columns))}")
    frame = frame[
        frame["dataset_index"].astype(int) % args.num_shards == args.shard_index
    ].sort_values("dataset_index").reset_index(drop=True)
    expected_shard_rows = args.expected_total_rows // args.num_shards
    if len(frame) != expected_shard_rows:
        raise RuntimeError(f"Expected {expected_shard_rows} shard rows, got {len(frame)}")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    prompts: list[TokensPrompt] = []
    metadata: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        message = deepmath_user_prompt(str(row.problem))
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": message}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
        if not prompt_ids or not all(isinstance(token_id, int) for token_id in prompt_ids):
            raise TypeError("Chat template did not return integer prompt token IDs")
        prompts.append(TokensPrompt(prompt_token_ids=list(prompt_ids)))
        metadata.append(
            {
                "dataset": "deepmath_ge6_test1000",
                "dataset_index": int(row.dataset_index),
                "problem_id": str(row.id),
                "problem": str(row.problem),
                "solution": str(row.solution),
                "difficulty": float(row.difficulty),
                "topic": str(row.topic),
                "source_index": int(row.source_index),
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
        n=args.num_rollouts,
        seed=args.seed,
    )
    started = time.time()
    outputs = engine.generate(prompts, sampling, use_tqdm=True)
    elapsed = time.time() - started

    rows: list[dict[str, Any]] = []
    for item, request_output in zip(metadata, outputs, strict=True):
        if len(request_output.outputs) != args.num_rollouts:
            raise RuntimeError(
                f"dataset_index={item['dataset_index']}: expected {args.num_rollouts} outputs, "
                f"got {len(request_output.outputs)}"
            )
        for replicate_index, generated in enumerate(request_output.outputs):
            response_ids = [int(value) for value in generated.token_ids]
            response = tokenizer.decode(response_ids, skip_special_tokens=False)
            row = {
                    **item,
                    "replicate_index": replicate_index,
                    "response_token_ids": response_ids,
                    "response": response,
                    "response_tokens": len(response_ids),
                    "finish_reason": generated.finish_reason,
                    "stop_reason": generated.stop_reason,
                    "target_answer": item["solution"],
                    "thinking_open_tag": response.count("<think>"),
                    "thinking_close_tag": response.count("</think>"),
                }
            if not args.generate_only:
                if "<think>" in response or "</think>" in response:
                    raise RuntimeError(
                        f"Thinking tag at dataset_index={item['dataset_index']} "
                        f"replicate={replicate_index}"
                    )
                correct, predicted, score_method = score_math_response(
                    response, item["solution"], args.score_timeout
                )
                row.update(
                    correct=bool(correct),
                    predicted_answer=predicted,
                    score_method=score_method,
                )
            rows.append(row)

    expected_outputs = expected_shard_rows * args.num_rollouts
    if len(rows) != expected_outputs:
        raise RuntimeError(f"Expected {expected_outputs} outputs, got {len(rows)}")
    args.output_dir.mkdir(parents=True)
    rollout_path = args.output_dir / "rollouts.jsonl"
    write_jsonl(rollout_path, rows)
    config = {
        "status": (
            "DEEPMATH_CHECKPOINT_NATIVE_NONTHINKING_X8_GENERATION_SHARD_OK"
            if args.generate_only
            else "DEEPMATH_CHECKPOINT_NATIVE_NONTHINKING_X8_SHARD_OK"
        ),
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "output_sha256": sha256(rollout_path),
        "student": args.model,
        "tokenizer": args.tokenizer,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "num_prompts": len(metadata),
        "num_rollouts": len(rows),
        "thinking_mode": False,
        "generation_only": bool(args.generate_only),
        "parameters_frozen": True,
        "optimizer_created": False,
        "backward_called": False,
        "sampling": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "n": args.num_rollouts,
            "seed": args.seed,
        },
        "elapsed_seconds_generate": elapsed,
        "total_response_tokens": sum(row["response_tokens"] for row in rows),
        "thinking_open_tag": sum(row["thinking_open_tag"] for row in rows),
        "thinking_close_tag": sum(row["thinking_close_tag"] for row in rows),
        "vllm_port": os.environ.get("VLLM_PORT"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "worker_rank": os.environ.get("OPD_WORKER_RANK"),
        "node": os.environ.get("SLURMD_NODENAME"),
    }
    write_json(args.output_dir / "run_config.json", config)
    (args.output_dir / "SHARD_COMPLETE").touch()
    print(json.dumps(config, ensure_ascii=False, sort_keys=True))
    print(config["status"])


if __name__ == "__main__":
    main()
