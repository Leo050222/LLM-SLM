#!/usr/bin/env python3
"""Strict structural verification for bridge-only Student-Top16 RKL training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


OBJECTIVE = "bridge_student_top_16_renormalized_kl_student_to_teacher"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--expected-checkpoints", required=True)
    parser.add_argument("--expected-covered-bridges", type=int, required=True)
    parser.add_argument("--expected-train-rows", type=int, default=105952)
    parser.add_argument("--expected-selection", type=Path)
    parser.add_argument("--require-full-epoch", action="store_true")
    parser.add_argument("--require-selected-epoch", action="store_true")
    parser.add_argument("--verify-hf-load", action="store_true")
    args = parser.parse_args()
    checkpoints = [int(value) for value in args.expected_checkpoints.split(",")]

    run = read(args.run_dir / "run_manifest.json")
    result = read(args.run_dir / "final_result.json")
    expected_run = {
        "objective": OBJECTIVE,
        "loss_direction": "KL(student||teacher)",
        "loss_support_topk": 16,
        "loss_mask": "bridge_tokens_only",
        "conditioning": "anchor_prefix_token_ids + bridge teacher forcing",
        "trace_count": 13244,
        "bridges_per_trace": 8,
        "bridge_tokens": 256,
        "train_rows": args.expected_train_rows,
        "global_batch_size": 64,
        "student_world_size": 4,
        "total_steps": args.expected_steps,
        "save_steps": checkpoints,
        "max_model_len": 40960,
        "all_bridges_covered_once": args.require_full_epoch,
    }
    if args.expected_selection is not None:
        expected_run["selected_bridges_covered_once"] = args.require_selected_epoch
    for key, expected in expected_run.items():
        if run.get(key) != expected:
            raise RuntimeError(f"run manifest mismatch {key}: {run.get(key)} != {expected}")
    if not Path(run["bridge_completion_marker"]).is_file():
        raise FileNotFoundError(run["bridge_completion_marker"])
    if result.get("status") != "BRIDGE_TOP16_RKL_COMPLETE":
        raise RuntimeError(f"unexpected final status: {result.get('status')}")
    result_keys = [
        "objective", "loss_support_topk", "loss_mask", "trace_count",
        "bridges_per_trace", "bridge_tokens", "total_steps", "checkpoints",
        "all_bridges_covered_once",
    ]
    if args.expected_selection is not None:
        result_keys.append("selected_bridges_covered_once")
    for key in result_keys:
        expected = {
            "objective": OBJECTIVE,
            "loss_support_topk": 16,
            "loss_mask": "bridge_tokens_only",
            "trace_count": 13244,
            "bridges_per_trace": 8,
            "bridge_tokens": 256,
            "total_steps": args.expected_steps,
            "checkpoints": checkpoints,
            "all_bridges_covered_once": args.require_full_epoch,
            "selected_bridges_covered_once": args.require_selected_epoch,
        }[key]
        if result.get(key) != expected:
            raise RuntimeError(f"final result mismatch {key}: {result.get(key)} != {expected}")

    seen_samples: set[int] = set()
    selected_samples: set[int] | None = None
    if args.expected_selection is not None:
        selection = read(args.expected_selection)
        if selection.get("status") != "GOOD_BRIDGE_SELECTION_COMPLETE":
            raise RuntimeError("selection status mismatch")
        selected_samples = {int(value) for value in selection.get("sample_ids", [])}
        if len(selected_samples) != args.expected_train_rows:
            raise RuntimeError("selection row count mismatch")
        expected_selection_fields = {
            "training_scope": "good_bridges_only",
            "bridge_selection": str(args.expected_selection),
            "bridge_selection_sha256": sha256(args.expected_selection),
            "bridge_selection_status": selection["status"],
            "bridge_selection_definition": selection["definition"],
        }
        for key, expected in expected_selection_fields.items():
            if run.get(key) != expected:
                raise RuntimeError(f"run selection mismatch {key}: {run.get(key)} != {expected}")
    covered = 0
    for step in range(1, args.expected_steps + 1):
        summary = read(args.run_dir / "metrics" / f"step_{step:04d}.json")
        if summary.get("status") != "BRIDGE_TOP16_RKL_STEP_COMPLETE":
            raise RuntimeError(f"step {step}: invalid summary status")
        rollout = summary["rollout"]
        student = summary["student"]
        step_ids = [int(row["sample_id"]) for row in rollout["rows"]]
        if len(step_ids) != int(rollout["examples"]) or len(set(step_ids)) != len(step_ids):
            raise RuntimeError(f"step {step}: duplicate or missing sample ids")
        if seen_samples.intersection(step_ids):
            raise RuntimeError(f"step {step}: sample reused across steps")
        seen_samples.update(step_ids)
        if selected_samples is not None and not set(step_ids).issubset(selected_samples):
            raise RuntimeError(f"step {step}: contains bridge outside strict selection")
        covered += len(step_ids)
        if int(rollout["total_response_tokens"]) != len(step_ids) * 256:
            raise RuntimeError(f"step {step}: bridge token accounting mismatch")
        for key in (
            "mean_reverse_kl", "gradient_norm_before_clip",
            "parameter_checksum_delta",
        ):
            if not math.isfinite(float(student[key])):
                raise RuntimeError(f"step {step}: non-finite {key}")
        if float(student["mean_reverse_kl"]) < -1e-6:
            raise RuntimeError(f"step {step}: negative reverse KL")
        if float(student["parameter_checksum_delta"]) == 0.0:
            raise RuntimeError(f"step {step}: optimizer did not change parameters")
        if len(student["rank_metrics"]) != 4:
            raise RuntimeError(f"step {step}: missing student rank metrics")
        if sum(int(row["local_tokens"]) for row in student["rank_metrics"]) != int(
            student["global_bridge_tokens"]
        ):
            raise RuntimeError(f"step {step}: rank token accounting mismatch")
    if covered != args.expected_covered_bridges or len(seen_samples) != covered:
        raise RuntimeError(
            f"bridge coverage mismatch: covered={covered} expected={args.expected_covered_bridges}"
        )
    if args.require_full_epoch and seen_samples != set(range(105952)):
        raise RuntimeError("full epoch does not cover every bridge exactly once")
    if args.require_selected_epoch and seen_samples != selected_samples:
        raise RuntimeError("selected epoch does not cover every selected bridge exactly once")

    found = sorted(
        int(path.name.removeprefix("global_step_"))
        for path in args.checkpoint_root.glob("global_step_*")
        if path.is_dir()
    )
    if found != checkpoints:
        raise RuntimeError(f"checkpoint set mismatch: {found}")
    for step in checkpoints:
        actor = args.checkpoint_root / f"global_step_{step}" / "actor"
        for relative in (
            "model.pt", "optimizer.pt", "extra_state.pt", "huggingface/config.json"
        ):
            if not (actor / relative).is_file():
                raise FileNotFoundError(actor / relative)
    if args.verify_hf_load:
        import torch
        from transformers import AutoModelForCausalLM

        final_hf = args.checkpoint_root / f"global_step_{checkpoints[-1]}" / "actor" / "huggingface"
        model = AutoModelForCausalLM.from_pretrained(
            final_hf, local_files_only=True, dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        if getattr(model.config, "vocab_size", None) != 151936:
            raise RuntimeError("unexpected final model vocabulary")
        if sum(parameter.numel() for parameter in model.parameters()) <= 0:
            raise RuntimeError("final model has no parameters")
    print("BRIDGE_TOP16_RKL_VERIFY_OK")


if __name__ == "__main__":
    main()
