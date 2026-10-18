#!/usr/bin/env python3
"""Run one Qwen3 model greedily on the frozen hard_three prompt set."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path

import requests

ROOT = Path("/storage/yukaichengLab/wangxin/test_eiloeo")
SOURCE = ROOT / "opd_eval_hard/hard_three_20260911"
N = 1814


def rows() -> list[dict]:
    with (SOURCE / "prompts.jsonl").open() as f:
        out = [json.loads(line) for line in f]
    if len(out) != N:
        raise RuntimeError(f"expected {N} frozen prompts, found {len(out)}")
    return out


def post_completion(session, url, model, prompt_ids, max_tokens, fallback_prompt):
    body = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "stream": False,
    }
    response = session.post(url, json=body, timeout=7200)
    if response.status_code >= 400:
        body["prompt"] = fallback_prompt
        response = session.post(url, json=body, timeout=7200)
    response.raise_for_status()
    choice = response.json()["choices"][0]
    return str(choice.get("text", "")), str(choice.get("finish_reason", "stop"))


def run(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    out_root = Path(args.output_root)
    completed = out_root / "completed"
    completed.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    prompt_rows = rows()
    prompt_sha256 = hashlib.sha256((SOURCE / "prompts.jsonl").read_bytes()).hexdigest()

    def one(row: dict) -> None:
        dest = completed / f"{row['dataset_index']:05d}.json"
        if dest.exists():
            old = json.loads(dest.read_text())
            if old.get("prompt_sha256") == prompt_sha256 and old.get("status") == "COMPLETE":
                return
            dest.unlink()
        ids = list(map(int, row["prompt_token_ids"]))
        max_tokens = min(40264, 40960 - len(ids) - 1)
        if max_tokens <= 0:
            raise RuntimeError(f"prompt {row['dataset_index']} leaves no generation room")
        prompt_text = tok.decode(ids, skip_special_tokens=False)
        with requests.Session() as session:
            text, finish_reason = post_completion(
                session, args.api_url, args.served_model, ids, max_tokens, prompt_text
            )
        record = dict(
            row,
            status="COMPLETE",
            response=text,
            response_token_ids=tok.encode(text, add_special_tokens=False),
            finish_reason=finish_reason,
            prompt_sha256=prompt_sha256,
            model=args.served_model,
            generation_config={"temperature": 0.0, "top_p": 1.0, "n": 1},
            generated_at=time.time(),
        )
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False))
        tmp.replace(dest)

    workers = args.workers or min(64, os.cpu_count() or 8)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, row) for row in prompt_rows]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if done % 50 == 0 or done == N:
                print(f"GENERATED {done}/{N}", flush=True)
    manifest = {
        "status": "COMPLETE",
        "rows": N,
        "model": args.served_model,
        "model_path": args.model_path,
        "prompt_sha256": prompt_sha256,
        "generation": {"temperature": 0.0, "top_p": 1.0, "n": 1, "max_context": 40960},
    }
    (out_root / "ALL_GENERATION_COMPLETE.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
