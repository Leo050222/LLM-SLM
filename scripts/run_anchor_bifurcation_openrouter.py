#!/usr/bin/env python3
"""Resume-safe concurrent OpenRouter judge runner for anchor bifurcation inputs."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "anchor_verdict": {"enum": ["yes", "no", "uncertain"]},
        "reasoning_direction_changed": {"enum": ["yes", "no", "uncertain"]},
        "divergence_onset": {"enum": ["immediately_after_anchor", "within_256_span", "after_256", "none_or_stylistic", "unclear"]},
        "change_types": {"type": "array", "items": {"enum": ["error_correction", "changed_assumption", "new_subgoal", "changed_strategy", "changed_interpretation", "different_computation_only", "stylistic_only", "other", "none"]}},
        "persistence": {"enum": ["persistent", "reconverged", "local_only", "none", "unclear"]},
        "pre_anchor_reasoning_state": {"type": "string"},
        "path_a_post_anchor_direction": {"type": "string"},
        "path_b_post_anchor_direction": {"type": "string"},
        "evidence_from_path_a": {"type": "string"},
        "evidence_from_path_b": {"type": "string"},
        "rationale": {"type": "string"},
        "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": ["anchor_verdict", "reasoning_direction_changed", "divergence_onset", "change_types", "persistence", "pre_anchor_reasoning_state", "path_a_post_anchor_direction", "path_b_post_anchor_direction", "evidence_from_path_a", "evidence_from_path_b", "rationale", "confidence"],
}


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def call(
    item: dict[str, Any],
    output_dir: Path,
    model: str,
    timeout: float,
    attempts: int,
    provider_order: list[str],
    allow_fallbacks: bool,
    api_url: str,
    api_key: str,
) -> str:
    destination = output_dir / "items" / f"{item['item_id']}.json"
    if destination.is_file():
        existing = json.loads(destination.read_text())
        if existing.get("status") == "ok" and existing.get("prompt_sha256") == item["prompt_sha256"] and existing.get("model") == model:
            return "reused"
    body = {
        "model": model,
        "messages": item["messages"],
        "temperature": 0,
        "max_tokens": 1200,
        "response_format": {"type": "json_schema", "json_schema": {"name": "anchor_bifurcation_judgment", "strict": True, "schema": SCHEMA}},
    }
    if provider_order:
        body["provider"] = {
            "order": provider_order,
            "allow_fallbacks": allow_fallbacks,
            "require_parameters": True,
        }
    encoded = json.dumps(body).encode()
    last_error = ""
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            api_url,
            data=encoded,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = json.loads(response.read())
            content = raw["choices"][0]["message"]["content"]
            judgment = json.loads(content)
            missing = set(SCHEMA["required"]) - set(judgment)
            if missing:
                raise RuntimeError(f"missing fields: {sorted(missing)}")
            record = {k: v for k, v in item.items() if k != "messages"}
            record.update({"status": "ok", "model": model, "judgment": judgment, "usage": raw.get("usage"), "provider": raw.get("provider"), "response_id": raw.get("id")})
            atomic_json(destination, record)
            destination.with_suffix(".failed.json").unlink(missing_ok=True)
            return "ok"
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            last_error = f"HTTPError {exc.code}: {detail}"
            retry_after = exc.headers.get("Retry-After")
            if attempt < attempts:
                try:
                    delay = float(retry_after) if retry_after else 2 ** attempt + random.random()
                except ValueError:
                    delay = 2 ** attempt + random.random()
                time.sleep(min(60.0, max(0.0, delay)))
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < attempts:
                time.sleep(min(60.0, 2 ** attempt + random.random()))
    atomic_json(destination.with_suffix(".failed.json"), {"status": "failed", "item_id": item["item_id"], "prompt_sha256": item["prompt_sha256"], "model": model, "error": last_error})
    return "failed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="openai/gpt-5.6-luna")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--provider-order", action="append", default=[])
    parser.add_argument("--allow-provider-fallbacks", action="store_true")
    parser.add_argument("--api-url", default="https://openrouter.ai/api/v1/chat/completions")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    args = parser.parse_args()
    api_key = os.environ.pop(args.api_key_env, None)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")
    items = [json.loads(line) for line in args.inputs.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        items = items[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "items").mkdir(exist_ok=True)
    started = time.time()
    counts: dict[str, int] = {"ok": 0, "reused": 0, "failed": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                call,
                item,
                args.output_dir,
                args.model,
                args.timeout,
                args.attempts,
                args.provider_order,
                args.allow_provider_fallbacks,
                args.api_url,
                api_key,
            )
            for item in items
        ]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            status = future.result()
            counts[status] += 1
            if completed % 25 == 0 or completed == len(futures):
                print(json.dumps({"completed": completed, "total": len(futures), **counts}), flush=True)
    backend = "OPENLUX" if "openlux.ai" in args.api_url else "OPENROUTER"
    summary = {"status": f"ANCHOR_BIFURCATION_{backend}_RUN_COMPLETE" if counts["failed"] == 0 else f"ANCHOR_BIFURCATION_{backend}_RUN_INCOMPLETE", "model": args.model, "api_url": args.api_url, "provider_order": args.provider_order, "allow_provider_fallbacks": args.allow_provider_fallbacks, "inputs": str(args.inputs), "inputs_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(), "requested": len(items), **counts, "elapsed_seconds": time.time() - started}
    atomic_json(args.output_dir / "run_summary.json", summary)
    if counts["failed"]:
        raise RuntimeError(f"{counts['failed']} OpenRouter requests failed")
    (args.output_dir / summary["status"]).touch()
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
