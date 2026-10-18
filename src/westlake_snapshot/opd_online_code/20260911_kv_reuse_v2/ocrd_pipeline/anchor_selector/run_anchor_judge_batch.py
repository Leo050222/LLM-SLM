#!/usr/bin/env python3
"""Judge shard candidate records concurrently and merge one result per trace."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR.parent))

from anchor_selector.core import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    decoded_prefix,
    read_jsonl,
    selected_anchor_artifact,
)
from anchor_selector.openlux_client import (  # noqa: E402
    OpenLuxClient,
    build_user_prompt,
    load_env_file,
)


class BillingUnavailable(RuntimeError):
    """Raised once the provider reports that paid requests cannot continue."""


_BILLING_ERROR_MARKERS = (
    "http 402",
    "insufficient_quota",
    "insufficient quota",
    "insufficient balance",
    "credit balance",
    "billing error",
    "billing_not_active",
    "余额不足",
    "额度不足",
    "欠费",
    "请充值",
)


def is_billing_error(error: BaseException) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in _BILLING_ERROR_MARKERS)


def load_api_key(config: dict[str, Any], secrets: Path) -> str:
    key_env = str(config.get("api_key_env", "OPENLUX_API_KEY"))
    key = os.environ.get(key_env, "")
    if not key:
        if secrets.is_file():
            key = load_env_file(secrets).get(key_env, "")
    if not key and not bool(config.get("api_key_required", True)):
        return str(config.get("api_key", "EMPTY"))
    if not key:
        raise ValueError(f"{key_env} missing from environment and {secrets}")
    return key


def discover_records(root: Path) -> list[Path]:
    paths = sorted(root.glob("shards/shard_*/candidate_records/trace_*.json"))
    if not paths:
        paths = sorted(root.glob("candidate_records/trace_*.json"))
    if not paths:
        raise FileNotFoundError(f"No candidate records found beneath {root}")
    return paths


def result_dir(output_dir: Path, dataset_index: int) -> Path:
    return output_dir / "traces" / f"trace_{dataset_index:05d}"


class Runtime:
    def __init__(
        self,
        config: dict[str, Any],
        api_key: str,
        tokenizer: Any,
        prefix_max_chars: int,
    ) -> None:
        self.config = config
        self.api_key = api_key
        self.tokenizer = tokenizer
        self.prefix_max_chars = prefix_max_chars
        self.local = threading.local()
        self.tokenizer_lock = threading.Lock()
        configured_urls = config.get("api_urls")
        if configured_urls is None:
            configured_urls = [config["api_url"]]
        if not isinstance(configured_urls, list) or not configured_urls:
            raise ValueError("api_urls must be a non-empty list when provided")
        self.api_urls = [str(value) for value in configured_urls]
        self.client_index = 0
        self.client_index_lock = threading.Lock()
        self.billing_stop_event = threading.Event()

    def client(self) -> OpenLuxClient:
        client = getattr(self.local, "client", None)
        if client is None:
            with self.client_index_lock:
                api_url = self.api_urls[self.client_index % len(self.api_urls)]
                self.client_index += 1
            client = OpenLuxClient(
                api_url=api_url,
                api_key=self.api_key,
                model=str(self.config["judge_model"]),
                temperature=float(self.config["judge_temperature"]),
                max_tokens=int(self.config["judge_max_tokens"]),
                timeout_seconds=float(self.config["judge_timeout_seconds"]),
                max_retries=int(self.config["judge_max_retries"]),
                reasoning_effort=self.config.get("judge_reasoning_effort"),
                extra_body=self.config.get("judge_extra_body"),
            )
            self.local.client = client
        return client

    def prefix(self, trace: dict[str, Any], position: int) -> tuple[str, bool, int]:
        # The Rust-backed fast tokenizer is generally thread safe, but keeping
        # decode under a short lock makes the batch runner deterministic across
        # tokenizer versions.
        with self.tokenizer_lock:
            return decoded_prefix(
                trace, self.tokenizer, position, self.prefix_max_chars
            )


def cached_final(path: Path, trace_sha256: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if value.get("trace_sha256") != trace_sha256:
        return None
    if value.get("status") not in {"ANCHOR_SELECTED", "NO_ANCHOR_SELECTED"}:
        return None
    return value


def enriched_selected(
    record: dict[str, Any],
    candidate: dict[str, Any],
    judgment: dict[str, Any],
) -> dict[str, Any]:
    artifact = selected_anchor_artifact(record["trace"], candidate, judgment)
    artifact.update(
        {
            "difficulty": record.get("difficulty"),
            "topic": record.get("topic"),
            "candidate_count": int(record["candidate_count"]),
            "selected_candidate": candidate,
        }
    )
    return artifact


def judge_candidate(
    record: dict[str, Any], candidate: dict[str, Any], runtime: Runtime
) -> dict[str, Any]:
    """Issue one candidate request; safe to run in the shared API executor."""
    stop_event = getattr(runtime, "billing_stop_event", None)
    if stop_event is not None and stop_event.is_set():
        raise BillingUnavailable("billing circuit is already open")
    position = int(candidate["response_position"])
    prefix_text, truncated, original_chars = runtime.prefix(record["trace"], position)
    prompt = build_user_prompt(
        record["trace"], candidate, prefix_text, truncated, original_chars
    )
    try:
        judgment, api_metadata = runtime.client().judge(prompt)
    except Exception as error:
        if is_billing_error(error):
            if stop_event is not None:
                stop_event.set()
            raise BillingUnavailable(str(error)) from error
        raise
    return {
        "response_position": position,
        "candidate_index": int(candidate["candidate_index"]),
        "absolute_logprob_gap": float(candidate["absolute_logprob_gap"]),
        "judgment": judgment,
        "api": api_metadata,
        "prefix_truncated": truncated,
        "prefix_original_chars": original_chars,
    }


def process_record(
    path: Path,
    output_dir: Path,
    runtime: Runtime,
    max_candidates: int,
    candidate_batch_size: int,
    api_executor: ThreadPoolExecutor,
) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "ANCHOR_CANDIDATES_OK":
        raise ValueError(f"Invalid candidate record: {path}")
    index = int(record["dataset_index"])
    work_dir = result_dir(output_dir, index)
    work_dir.mkdir(parents=True, exist_ok=True)
    final_path = work_dir / "selected_anchor.json"
    final = cached_final(final_path, str(record["trace_sha256"]))
    if final is not None:
        return {
            "dataset_index": index,
            "status": final["status"],
            "cached": True,
            "judge_calls": 0,
        }

    candidates = sorted(record["candidates"], key=lambda row: int(row["response_position"]))
    if max_candidates > 0:
        candidates = candidates[:max_candidates]
    judgments_path = work_dir / "judgments.jsonl"
    judgments = read_jsonl(judgments_path) if judgments_path.is_file() else []
    by_position = {int(row["response_position"]): row for row in judgments}
    selected: tuple[dict[str, Any], dict[str, Any]] | None = None
    new_calls = 0

    for batch_start in range(0, len(candidates), candidate_batch_size):
        batch = candidates[batch_start : batch_start + candidate_batch_size]
        missing = [
            candidate
            for candidate in batch
            if int(candidate["response_position"]) not in by_position
        ]
        future_to_candidate = {
            api_executor.submit(judge_candidate, record, candidate, runtime): candidate
            for candidate in missing
        }
        new_rows: list[dict[str, Any]] = []
        errors: list[BaseException] = []
        for future in as_completed(future_to_candidate):
            try:
                new_rows.append(future.result())
            except BaseException as error:
                errors.append(error)
        if new_rows:
            for row in new_rows:
                by_position[int(row["response_position"])] = row
            judgments.extend(new_rows)
            judgments.sort(key=lambda value: int(value["response_position"]))
            atomic_write_jsonl(judgments_path, judgments)
            new_calls += len(new_rows)
        if errors:
            # Preserve every valid response from the batch, but do not select
            # an anchor while any earlier candidate in this batch is unknown.
            billing_errors = [
                error for error in errors if isinstance(error, BillingUnavailable)
            ]
            raise billing_errors[0] if billing_errors else errors[0]

        accepted = [
            (candidate, by_position[int(candidate["response_position"])])
            for candidate in batch
            if bool(
                by_position[int(candidate["response_position"])]["judgment"][
                    "select_anchor"
                ]
            )
        ]
        if accepted:
            # Requests in the batch finish in arbitrary wall-clock order.  The
            # anchor definition is chronological, so choose the earliest yes.
            selected = min(
                accepted, key=lambda item: int(item[0]["response_position"])
            )
            break

    if selected is not None:
        final = enriched_selected(record, selected[0], selected[1])
    else:
        final = {
            "status": "NO_ANCHOR_SELECTED",
            "dataset_index": index,
            "problem_id": record.get("problem_id"),
            "trace_sha256": record["trace_sha256"],
            "prompt_tokens": int(record["prompt_tokens"]),
            "response_tokens": int(record["response_tokens"]),
            "difficulty": record.get("difficulty"),
            "topic": record.get("topic"),
            "candidate_count": int(record["candidate_count"]),
            "judged_count": len(judgments),
            "exhaustive": max_candidates <= 0 or max_candidates >= int(record["candidate_count"]),
        }
    atomic_write_json(final_path, final)
    return {
        "dataset_index": index,
        "status": final["status"],
        "cached": False,
        "judge_calls": new_calls,
    }


def usage_total(judgments: list[dict[str, Any]]) -> int:
    total = 0
    for row in judgments:
        usage = row.get("api", {}).get("usage") or {}
        value = usage.get("total_tokens")
        if isinstance(value, (int, float)):
            total += int(value)
    return total


def merge_results(
    records: list[Path],
    output_dir: Path,
    failures: list[dict[str, Any]],
    started: float,
    stop_reason: str | None = None,
) -> dict[str, Any]:
    finals: list[dict[str, Any]] = []
    all_judgments: list[dict[str, Any]] = []
    missing: list[int] = []
    for path in records:
        record = json.loads(path.read_text(encoding="utf-8"))
        index = int(record["dataset_index"])
        final_path = result_dir(output_dir, index) / "selected_anchor.json"
        final = cached_final(final_path, str(record["trace_sha256"]))
        if final is None:
            missing.append(index)
            continue
        finals.append(final)
        judgments_path = result_dir(output_dir, index) / "judgments.jsonl"
        if judgments_path.is_file():
            for judgment in read_jsonl(judgments_path):
                all_judgments.append({"dataset_index": index, **judgment})
    finals.sort(key=lambda row: int(row["dataset_index"]))
    all_judgments.sort(
        key=lambda row: (int(row["dataset_index"]), int(row["response_position"]))
    )
    atomic_write_jsonl(output_dir / "anchors.jsonl", finals)
    atomic_write_jsonl(output_dir / "judgments.jsonl", all_judgments)
    atomic_write_jsonl(output_dir / "failures.jsonl", failures)
    selected = [row for row in finals if row["status"] == "ANCHOR_SELECTED"]
    positions = [
        float(row["anchor_response_position"]) / max(int(row["response_tokens"]), 1)
        for row in selected
    ]
    labels = Counter(
        row.get("judge", {}).get("judgment", {}).get("label", "unknown")
        for row in selected
    )
    summary = {
        "status": "BATCH_JUDGE_COMPLETE" if not failures and not missing else "BATCH_JUDGE_INCOMPLETE",
        "expected_traces": len(records),
        "completed_traces": len(finals),
        "selected_anchors": len(selected),
        "no_anchor_selected": sum(row["status"] == "NO_ANCHOR_SELECTED" for row in finals),
        "failed_traces": len(failures),
        "missing_traces": missing,
        "judge_calls": len(all_judgments),
        "recorded_api_tokens": usage_total(all_judgments),
        "selected_labels": dict(sorted(labels.items())),
        "selected_relative_position_mean": statistics.fmean(positions) if positions else None,
        "selected_relative_position_median": statistics.median(positions) if positions else None,
        "elapsed_seconds": time.time() - started,
        "stop_reason": stop_reason,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    if summary["status"] == "BATCH_JUDGE_COMPLETE":
        (output_dir / "BATCH_JUDGE_COMPLETE").touch()
    if stop_reason == "billing_unavailable":
        (output_dir / "BILLING_STOPPED").touch()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--config", type=Path, default=MODULE_DIR / "config.json")
    parser.add_argument("--secrets", type=Path, default=MODULE_DIR / ".secrets.env")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=1,
        help="Concurrent chronological candidates per trace; choose earliest yes in each batch",
    )
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-candidates", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    from transformers import AutoTokenizer

    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.candidate_batch_size <= 0:
        raise ValueError("--candidate-batch-size must be positive")
    records = discover_records(args.input_root)
    if args.max_records is not None:
        if args.max_records <= 0:
            raise ValueError("--max-records must be positive")
        records = records[: args.max_records]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    runtime = Runtime(
        config,
        load_api_key(config, args.secrets),
        tokenizer,
        int(config["prefix_max_chars"]),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    failures: list[dict[str, Any]] = []
    completed = 0
    stop_reason: str | None = None
    api_workers = args.workers * args.candidate_batch_size
    with ThreadPoolExecutor(max_workers=api_workers) as api_executor:
        with ThreadPoolExecutor(max_workers=args.workers) as trace_executor:
            futures = {
                trace_executor.submit(
                    process_record,
                    path,
                    args.output_dir,
                    runtime,
                    args.max_candidates,
                    args.candidate_batch_size,
                    api_executor,
                ): path
                for path in records
            }
            for future in as_completed(futures):
                path = futures[future]
                completed += 1
                try:
                    result = future.result()
                    print(
                        f"JUDGE_PROGRESS traces={completed}/{len(records)} "
                        f"dataset_index={result['dataset_index']} status={result['status']} "
                        f"new_calls={result['judge_calls']}",
                        flush=True,
                    )
                except Exception as error:  # keep independent traces progressing
                    failure = {
                        "candidate_record": str(path),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                    failures.append(failure)
                    print(
                        f"JUDGE_FAILED traces={completed}/{len(records)} path={path} "
                        f"error={type(error).__name__}: {error}",
                        flush=True,
                    )
                    if isinstance(error, BillingUnavailable):
                        stop_reason = "billing_unavailable"
                        runtime.billing_stop_event.set()
                        for pending in futures:
                            pending.cancel()
                        print(
                            "BILLING_STOPPED cancelling_pending_traces=true",
                            flush=True,
                        )
                        break
    summary = merge_results(
        records, args.output_dir, failures, started, stop_reason=stop_reason
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    if stop_reason == "billing_unavailable":
        raise SystemExit(75)


if __name__ == "__main__":
    main()
