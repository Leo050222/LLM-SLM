#!/usr/bin/env python3
"""Fail closed unless student-top16 Reverse-KL training is structurally valid."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


OBJECTIVE = "student_top_16_renormalized_kl_student_to_teacher"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("--expected-steps", type=int, default=1)
    parser.add_argument("--expected-global-batch", type=int, default=16)
    parser.add_argument("--expected-checkpoints", default="1")
    parser.add_argument("--expected-student-world-size", type=int, default=8)
    parser.add_argument("--expected-max-prompt-tokens", type=int, default=695)
    parser.add_argument("--expected-thinking", choices=("true", "false"), default="true")
    parser.add_argument("--verify-hf-load", action="store_true")
    args = parser.parse_args()
    expected_checkpoints = [int(value) for value in args.expected_checkpoints.split(",")]

    run = read(args.root / "run_manifest.json")
    result = read(args.root / "final_result.json")
    expected = {
        "objective": OBJECTIVE,
        "loss_support_topk": 16,
        "student_world_size": args.expected_student_world_size,
        "global_batch_size": args.expected_global_batch,
        "total_steps": args.expected_steps,
        "max_model_len": 40960,
        "max_response_tokens": 40264,
        "max_prompt_tokens": args.expected_max_prompt_tokens,
        "thinking": args.expected_thinking == "true",
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
    }
    for key, value in expected.items():
        if run.get(key) != value:
            raise RuntimeError(f"Smoke manifest mismatch {key}: {run.get(key)} != {value}")
    expected_result = {
        "status": "DEEPMATH_RKL_COMPLETE",
        "objective": OBJECTIVE,
        "loss_support_topk": 16,
        "total_steps": args.expected_steps,
        "checkpoints": expected_checkpoints,
        "thinking": args.expected_thinking == "true",
    }
    if result != expected_result:
        raise RuntimeError(f"Unexpected training result: {result}")
    for step in range(1, args.expected_steps + 1):
        summary = read(args.root / "metrics" / f"step_{step:04d}.json")
        metric = summary["student"]
        rollout = summary["rollout"]
        for key in ("mean_reverse_kl", "gradient_norm_before_clip", "parameter_checksum_delta"):
            if not math.isfinite(float(metric[key])):
                raise RuntimeError(f"Non-finite smoke metric at step {step}: {key}")
        if float(metric["mean_reverse_kl"]) < -1e-6:
            raise RuntimeError("Restricted Reverse KL is unexpectedly negative")
        if float(metric["parameter_checksum_delta"]) == 0.0:
            raise RuntimeError("Smoke optimizer step did not change the model")
        if len(metric["rank_metrics"]) != args.expected_student_world_size:
            raise RuntimeError("Training did not return every student rank metric")
        if sum(row["local_tokens"] for row in metric["rank_metrics"]) != metric["global_response_tokens"]:
            raise RuntimeError("Smoke token accounting does not close")
        if args.expected_thinking == "false" and rollout.get(
            "generated_thinking_tag_count"
        ) != 0:
            raise RuntimeError(f"Thinking tags found in non-thinking step {step}")
    for step in expected_checkpoints:
        checkpoint = args.checkpoint_root / f"global_step_{step}" / "actor"
        for relative in (
            "model.pt", "optimizer.pt", "extra_state.pt", "huggingface/config.json"
        ):
            if not (checkpoint / relative).exists():
                raise FileNotFoundError(checkpoint / relative)
    if args.verify_hf_load:
        import torch
        from transformers import AutoModelForCausalLM

        final_hf = (
            args.checkpoint_root / f"global_step_{expected_checkpoints[-1]}" /
            "actor" / "huggingface"
        )
        model = AutoModelForCausalLM.from_pretrained(
            final_hf, local_files_only=True, dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        if getattr(model.config, "vocab_size", None) != 151936:
            raise RuntimeError(
                f"Unexpected final HF vocab size: {model.config.vocab_size}"
            )
        if sum(parameter.numel() for parameter in model.parameters()) <= 0:
            raise RuntimeError("Final HF model has no parameters")
    print("DEEPMATH_TOP16_TRAIN_VERIFY_OK")


if __name__ == "__main__":
    main()
