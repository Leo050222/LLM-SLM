#!/usr/bin/env python3
"""Score baseline student-sampled tokens under frozen student or teacher."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


CHUNK_TOKENS = 256


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


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


def score(args: argparse.Namespace) -> None:
    import torch

    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    rows = sorted(read_jsonl(args.rollouts), key=lambda row: int(row["dataset_index"]))
    model_path = args.student if args.phase == "student" else args.teacher
    model = load_model(model_path)
    score_dir = args.output_dir / f"{args.phase}_logprobs"
    score_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    started = time.monotonic()

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in rows:
            index = int(row["dataset_index"])
            response_ids = row["response_token_ids"]
            response_tokens = int(row["response_tokens"])
            if len(response_ids) != response_tokens:
                raise RuntimeError(f"index={index}: response token accounting mismatch")
            output = score_dir / f"example_{index:04d}.npz"
            if output.is_file():
                with np.load(output) as cached:
                    if cached["sample_logprob"].shape == (response_tokens,):
                        manifest.append({"dataset_index": index, "response_tokens": response_tokens, "cached": True})
                        continue

            values = row["prompt_token_ids"] + response_ids
            input_ids = torch.tensor(values, dtype=torch.long, device="cuda").unsqueeze(0)
            hidden = model.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                return_dict=True,
            ).last_hidden_state[0]
            position0 = len(row["prompt_token_ids"]) - 1
            scored: list[np.ndarray] = []
            for start in range(0, response_tokens, CHUNK_TOKENS):
                end = min(start + CHUNK_TOKENS, response_tokens)
                logits = model.lm_head(hidden[position0 + start : position0 + end]).float()
                sampled = torch.tensor(response_ids[start:end], dtype=torch.long, device="cuda")
                logprob = logits.gather(1, sampled[:, None]).squeeze(1) - torch.logsumexp(logits, dim=-1)
                scored.append(logprob.cpu().numpy().astype(np.float32))
                del logits, sampled, logprob
            values_np = np.concatenate(scored)
            if values_np.shape != (response_tokens,) or not np.isfinite(values_np).all():
                raise RuntimeError(f"index={index}: invalid log-probability output")
            temporary = output.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, sample_logprob=values_np)
            temporary.replace(output)
            manifest.append({
                "dataset_index": index,
                "response_tokens": response_tokens,
                "mean_logprob": float(values_np.mean()),
                "cached": False,
            })
            del input_ids, hidden, scored, values_np
            gc.collect()
            torch.cuda.empty_cache()

    write_json(args.output_dir / f"{args.phase}_manifest.json", {
        "status": f"DEEPMATH_BASELINE_{args.phase.upper()}_SAMPLED_LOGPROB_OK",
        "model": model_path,
        "rows": len(manifest),
        "tokens": sum(int(item["response_tokens"]) for item in manifest),
        "definition": "log pi_model(y_t | s_t) for the actual baseline student-sampled token",
        "elapsed_seconds": time.monotonic() - started,
        "items": manifest,
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["student", "teacher"])
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score(args)
