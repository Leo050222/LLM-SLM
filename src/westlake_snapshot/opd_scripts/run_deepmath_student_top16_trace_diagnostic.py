#!/usr/bin/env python3
"""Compare frozen Qwen3-1.7B student Top-16 support with Qwen3-14B.

The input responses are existing Qwen3-1.7B rollouts. Only rows where the
student answer is wrong and the independent 14B answer is correct are scored.
Student support selection and teacher scoring are separate phases so only one
model occupies the GPU at a time.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


TOPK = 16
CHUNK_TOKENS = 256


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def selected_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    teacher = {
        int(row["dataset_index"]): row
        for row in read_jsonl(args.teacher_rollouts)
    }
    rows = read_jsonl(args.rollouts)
    selected = [
        row
        for row in rows
        if not bool(row["correct"])
        and bool(teacher[int(row["dataset_index"])]["correct"])
    ]
    return sorted(selected, key=lambda row: int(row["dataset_index"]))


def load_model(path: str) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def phase_student(args: argparse.Namespace) -> None:
    import torch

    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = load_model(args.student)
    support_dir = args.output_dir / "student_support"
    support_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    started = time.monotonic()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in selected_rows(args):
            index = int(row["dataset_index"])
            output = support_dir / f"example_{index:03d}.npz"
            if output.is_file():
                with np.load(output) as cached:
                    if cached["student_top16_ids"].shape == (int(row["response_tokens"]), TOPK):
                        manifest.append({"dataset_index": index, "cached": True})
                        continue
            values = row["prompt_token_ids"] + row["response_token_ids"]
            input_ids = torch.tensor(values, dtype=torch.long, device="cuda").unsqueeze(0)
            hidden = model.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                return_dict=True,
            ).last_hidden_state[0]
            position0 = len(row["prompt_token_ids"]) - 1
            response_tokens = len(row["response_token_ids"])
            selected_ids: list[np.ndarray] = []
            selected_logp: list[np.ndarray] = []
            sampled_logp: list[np.ndarray] = []
            for start in range(0, response_tokens, CHUNK_TOKENS):
                end = min(start + CHUNK_TOKENS, response_tokens)
                logits = model.lm_head(hidden[position0 + start : position0 + end]).float()
                log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
                top_logits, top_ids = torch.topk(logits, k=TOPK, dim=-1)
                sampled_ids = torch.tensor(
                    row["response_token_ids"][start:end], dtype=torch.long, device="cuda"
                )
                selected_ids.append(top_ids.cpu().numpy().astype(np.int32))
                selected_logp.append((top_logits - log_z).cpu().numpy().astype(np.float32))
                sampled_logp.append(
                    (logits.gather(1, sampled_ids[:, None]).squeeze(1) - log_z.squeeze(1))
                    .cpu().numpy().astype(np.float32)
                )
                del logits, log_z, top_logits, top_ids, sampled_ids
            temporary = output.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    student_top16_ids=np.concatenate(selected_ids),
                    student_top16_logprobs=np.concatenate(selected_logp),
                    student_sample_logprob=np.concatenate(sampled_logp),
                )
            temporary.replace(output)
            manifest.append({
                "dataset_index": index,
                "response_tokens": response_tokens,
                "bytes": output.stat().st_size,
                "cached": False,
            })
            del input_ids, hidden, selected_ids, selected_logp, sampled_logp
            gc.collect()
            torch.cuda.empty_cache()
    write_json(args.output_dir / "student_manifest.json", {
        "status": "STUDENT_TOP16_SUPPORT_OK",
        "topk": TOPK,
        "selected_rows": len(manifest),
        "elapsed_seconds": time.monotonic() - started,
        "rows": manifest,
    })


def phase_teacher(args: argparse.Namespace) -> None:
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = load_model(args.teacher)
    tokenizer = AutoTokenizer.from_pretrained(
        args.student, local_files_only=True, trust_remote_code=False
    )
    token_dir = args.output_dir / "token_shards"
    detail_dir = args.output_dir / "top16_details"
    token_dir.mkdir(parents=True, exist_ok=True)
    detail_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    started = time.monotonic()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in selected_rows(args):
            index = int(row["dataset_index"])
            parquet = token_dir / f"example_{index:03d}.parquet"
            detail = detail_dir / f"example_{index:03d}.npz"
            if parquet.is_file() and detail.is_file():
                manifest.append({"dataset_index": index, "cached": True})
                continue
            with np.load(args.output_dir / "student_support" / f"example_{index:03d}.npz") as support:
                student_ids_np = support["student_top16_ids"].astype(np.int64)
                student_lp_np = support["student_top16_logprobs"].astype(np.float32)
                student_sample_np = support["student_sample_logprob"].astype(np.float32)
            values = row["prompt_token_ids"] + row["response_token_ids"]
            input_ids = torch.tensor(values, dtype=torch.long, device="cuda").unsqueeze(0)
            hidden = model.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                return_dict=True,
            ).last_hidden_state[0]
            position0 = len(row["prompt_token_ids"]) - 1
            response_tokens = len(row["response_token_ids"])
            teacher_support_lp: list[np.ndarray] = []
            teacher_sample_lp: list[np.ndarray] = []
            support_kl: list[np.ndarray] = []
            overlap: list[np.ndarray] = []
            top1: list[np.ndarray] = []
            for start in range(0, response_tokens, CHUNK_TOKENS):
                end = min(start + CHUNK_TOKENS, response_tokens)
                logits = model.lm_head(hidden[position0 + start : position0 + end]).float()
                log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
                support_ids = torch.from_numpy(student_ids_np[start:end]).to("cuda")
                teacher_selected_logits = logits.gather(1, support_ids)
                teacher_selected_lp = teacher_selected_logits - log_z
                student_selected_lp = torch.from_numpy(student_lp_np[start:end]).to("cuda")
                student_renorm = F.log_softmax(student_selected_lp, dim=-1)
                teacher_renorm = F.log_softmax(teacher_selected_lp, dim=-1)
                kl = (student_renorm.exp() * (student_renorm - teacher_renorm)).sum(dim=-1)
                teacher_top_ids = torch.topk(logits, k=TOPK, dim=-1).indices
                shared = (
                    support_ids.unsqueeze(-1) == teacher_top_ids.unsqueeze(-2)
                ).any(dim=-1).sum(dim=-1).float() / TOPK
                sampled_ids = torch.tensor(
                    row["response_token_ids"][start:end], dtype=torch.long, device="cuda"
                )
                teacher_sample = logits.gather(1, sampled_ids[:, None]).squeeze(1) - log_z.squeeze(1)
                teacher_support_lp.append(teacher_selected_lp.cpu().numpy().astype(np.float32))
                teacher_sample_lp.append(teacher_sample.cpu().numpy().astype(np.float32))
                support_kl.append(kl.cpu().numpy().astype(np.float32))
                overlap.append(shared.cpu().numpy().astype(np.float32))
                top1.append((support_ids[:, 0] == teacher_top_ids[:, 0]).cpu().numpy().astype(np.uint8))
                del logits, log_z, support_ids, teacher_selected_logits, teacher_selected_lp
                del student_selected_lp, student_renorm, teacher_renorm, kl, teacher_top_ids
                del shared, sampled_ids, teacher_sample
            teacher_support_np = np.concatenate(teacher_support_lp)
            teacher_sample_np = np.concatenate(teacher_sample_lp)
            kl_np = np.concatenate(support_kl)
            overlap_np = np.concatenate(overlap)
            top1_np = np.concatenate(top1)
            gap = teacher_sample_np - student_sample_np
            k1 = -gap
            k3 = np.expm1(np.minimum(gap, 80.0)) - gap
            response_ids = np.asarray(row["response_token_ids"], dtype=np.int32)
            pieces = tokenizer.convert_ids_to_tokens(response_ids.tolist())
            texts = [tokenizer.decode([int(token_id)], skip_special_tokens=False) for token_id in response_ids]
            think_end_id = tokenizer.convert_tokens_to_ids("</think>")
            matches = np.flatnonzero(response_ids == think_end_id)
            think_end = int(matches[0]) if matches.size else response_tokens
            frame = pd.DataFrame({
                "dataset_index": index,
                "problem_id": str(row["problem_id"]),
                "difficulty": row.get("difficulty"),
                "topic": row.get("topic"),
                "correct": bool(row["correct"]),
                "response_position": np.arange(response_tokens, dtype=np.int32),
                "relative_position": np.arange(response_tokens, dtype=np.float32) / max(1, response_tokens),
                "token_id": response_ids,
                "token_piece": pieces,
                "token_text": texts,
                "region": np.where(np.arange(response_tokens) <= think_end, "thinking", "final"),
                "student_top16_renorm_kl": kl_np,
                "student_teacher_top16_overlap": overlap_np,
                "top1_agreement": top1_np,
                "student_sample_logprob": student_sample_np,
                "teacher_sample_logprob": teacher_sample_np,
                "sample_logprob_gap_student_minus_teacher": k1,
                "sampled_reverse_kl_k1": k1,
                "sampled_reverse_kl_k3": k3,
            })
            frame.to_parquet(parquet, index=False, compression="zstd")
            temporary = detail.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    student_top16_ids=student_ids_np.astype(np.int32),
                    student_top16_logprobs=student_lp_np,
                    teacher_logprobs_on_student_top16=teacher_support_np,
                )
            temporary.replace(detail)
            manifest.append({
                "dataset_index": index,
                "response_tokens": response_tokens,
                "parquet_bytes": parquet.stat().st_size,
                "detail_bytes": detail.stat().st_size,
                "cached": False,
            })
            del input_ids, hidden, teacher_support_lp, teacher_sample_lp, support_kl, overlap, top1
            del teacher_support_np, teacher_sample_np, kl_np, overlap_np, top1_np, gap, k1, k3, frame
            gc.collect()
            torch.cuda.empty_cache()
    write_json(args.output_dir / "teacher_manifest.json", {
        "status": "TEACHER_ON_STUDENT_TOP16_OK",
        "topk": TOPK,
        "selected_rows": len(manifest),
        "metric": "renormalized KL(student || teacher) on student-selected Top-16 support",
        "elapsed_seconds": time.monotonic() - started,
        "rows": manifest,
    })


def phase_summarize(args: argparse.Namespace) -> None:
    import pandas as pd

    shards = sorted((args.output_dir / "token_shards").glob("example_*.parquet"))
    if not shards:
        raise FileNotFoundError("No token shards")
    frame = pd.concat((pd.read_parquet(path) for path in shards), ignore_index=True)
    metrics = [
        "student_top16_renorm_kl",
        "student_teacher_top16_overlap",
        "top1_agreement",
        "student_sample_logprob",
        "teacher_sample_logprob",
        "sample_logprob_gap_student_minus_teacher",
        "sampled_reverse_kl_k3",
    ]
    per_rollout = frame.groupby("dataset_index", observed=True)[metrics].mean().reset_index()
    frame.to_parquet(args.output_dir / "tokens.parquet", index=False, compression="zstd")
    per_rollout.to_parquet(args.output_dir / "per_rollout.parquet", index=False, compression="zstd")
    write_json(args.output_dir / "summary.json", {
        "status": "STUDENT_TOP16_TRACE_DIAGNOSTIC_OK",
        "topk": TOPK,
        "selected_rollouts": int(frame["dataset_index"].nunique()),
        "tokens": int(len(frame)),
        "metric": "renormalized KL(student || teacher) on student-selected Top-16 support",
        "token_weighted_mean": {name: float(frame[name].mean()) for name in metrics},
        "rollout_weighted_mean": {name: float(per_rollout[name].mean()) for name in metrics},
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["student", "teacher", "summarize"])
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--teacher-rollouts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    {"student": phase_student, "teacher": phase_teacher, "summarize": phase_summarize}[args.phase](args)


if __name__ == "__main__":
    main()
