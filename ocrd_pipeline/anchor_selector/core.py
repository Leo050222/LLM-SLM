"""Core data handling and candidate construction for anchor selection."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REQUIRED_TRACE_FIELDS = ("prompt_token_ids", "response_token_ids")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _validate_trace(trace: dict[str, Any]) -> dict[str, Any]:
    for field in REQUIRED_TRACE_FIELDS:
        if field not in trace:
            raise KeyError(f"Trace is missing required field: {field}")
        values = trace[field]
        if not isinstance(values, list) or not values:
            raise ValueError(f"Trace field {field} must be a non-empty list")
        if not all(isinstance(value, int) and value >= 0 for value in values):
            raise TypeError(f"Trace field {field} must contain non-negative integers")
    return trace


def load_trace(
    path: Path,
    row_index: int = 0,
    dataset_index: int | None = None,
    problem_id: str | None = None,
) -> dict[str, Any]:
    """Load one trace from JSON or JSONL using an explicit selector."""
    if path.suffix == ".jsonl":
        rows = read_jsonl(path)
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value if isinstance(value, list) else [value]
    if not rows:
        raise ValueError(f"No traces found in {path}")

    if dataset_index is not None:
        matched = [row for row in rows if int(row.get("dataset_index", -1)) == dataset_index]
        if len(matched) != 1:
            raise ValueError(
                f"Expected exactly one trace with dataset_index={dataset_index}, "
                f"found {len(matched)}"
            )
        return _validate_trace(matched[0])
    if problem_id is not None:
        matched = [row for row in rows if str(row.get("problem_id")) == problem_id]
        if len(matched) != 1:
            raise ValueError(
                f"Expected exactly one trace with problem_id={problem_id!r}, "
                f"found {len(matched)}"
            )
        return _validate_trace(matched[0])
    if not 0 <= row_index < len(rows):
        raise IndexError(f"row_index={row_index} outside [0, {len(rows)})")
    return _validate_trace(rows[row_index])


def trace_identity(trace: dict[str, Any]) -> dict[str, Any]:
    digest = hashlib.sha256()
    for token_id in trace["prompt_token_ids"] + trace["response_token_ids"]:
        digest.update(int(token_id).to_bytes(4, "little", signed=False))
    return {
        "dataset_index": trace.get("dataset_index"),
        "problem_id": trace.get("problem_id"),
        "trace_sha256": digest.hexdigest(),
        "prompt_tokens": len(trace["prompt_token_ids"]),
        "response_tokens": len(trace["response_token_ids"]),
    }


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """Hash the complete token-to-ID map, not only the vocabulary size."""
    digest = hashlib.sha256()
    ordered = sorted(tokenizer.get_vocab().items(), key=lambda item: (item[1], item[0]))
    for token, token_id in ordered:
        digest.update(int(token_id).to_bytes(4, "little", signed=False))
        digest.update(token.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def logprob_value(value: Any) -> float:
    if hasattr(value, "logprob"):
        return float(value.logprob)
    if isinstance(value, dict) and "logprob" in value:
        return float(value["logprob"])
    return float(value)


def lookup_actual_logprob(entry: Any, token_id: int) -> float:
    if entry is None:
        raise ValueError("Missing prompt-logprob entry")
    value = entry.get(token_id)
    if value is None:
        value = entry.get(str(token_id))
    if value is None:
        raise KeyError(f"Actual token {token_id} missing from prompt logprobs")
    result = logprob_value(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite logprob for token {token_id}: {result}")
    return result


def sorted_topk(entry: Any, k: int, vocab_size: int) -> tuple[list[int], list[float]]:
    values: list[tuple[float, int]] = []
    for raw_token_id, raw_value in entry.items():
        token_id = int(raw_token_id)
        if not 0 <= token_id < vocab_size:
            continue
        logprob = logprob_value(raw_value)
        if math.isfinite(logprob):
            values.append((logprob, token_id))
    values.sort(key=lambda item: (-item[0], item[1]))
    selected = values[:k]
    if len(selected) != k:
        raise ValueError(f"Expected {k} top tokens, found {len(selected)}")
    return [token_id for _, token_id in selected], [logprob for logprob, _ in selected]


def probability(logprob: float) -> float:
    if logprob < -745.0:
        return 0.0
    return math.exp(logprob)


def build_candidates(
    trace: dict[str, Any],
    tokenizer: Any,
    student_chosen_logprobs: np.ndarray,
    teacher_chosen_logprobs: np.ndarray,
    teacher_topk_ids: np.ndarray,
    teacher_topk_logprobs: np.ndarray,
    threshold: float = 2.0,
) -> list[dict[str, Any]]:
    """Return chronological candidates satisfying strict |g_t| > threshold."""
    response_ids = trace["response_token_ids"]
    expected = len(response_ids)
    arrays = {
        "student_chosen_logprobs": student_chosen_logprobs,
        "teacher_chosen_logprobs": teacher_chosen_logprobs,
        "teacher_topk_ids": teacher_topk_ids,
        "teacher_topk_logprobs": teacher_topk_logprobs,
    }
    for name, values in arrays.items():
        if values.shape[0] != expected:
            raise ValueError(f"{name} has {values.shape[0]} rows; expected {expected}")
    if teacher_topk_ids.shape != teacher_topk_logprobs.shape:
        raise ValueError("Teacher top-k IDs/logprobs shape mismatch")

    candidates: list[dict[str, Any]] = []
    for position, token_id in enumerate(response_ids):
        student_lp = float(student_chosen_logprobs[position])
        teacher_lp = float(teacher_chosen_logprobs[position])
        signed_gap = student_lp - teacher_lp
        absolute_gap = abs(signed_gap)
        if not absolute_gap > threshold:
            continue
        top_tokens: list[dict[str, Any]] = []
        for rank, (top_id, top_lp) in enumerate(
            zip(
                teacher_topk_ids[position].tolist(),
                teacher_topk_logprobs[position].tolist(),
                strict=True,
            ),
            start=1,
        ):
            top_tokens.append(
                {
                    "rank": rank,
                    "token_id": int(top_id),
                    "token_piece": tokenizer.convert_ids_to_tokens(int(top_id)),
                    "token_text": tokenizer.decode(
                        [int(top_id)], skip_special_tokens=False
                    ),
                    "logprob": float(top_lp),
                    "probability": probability(float(top_lp)),
                }
            )
        candidates.append(
            {
                "candidate_index": len(candidates),
                "response_position": position,
                "absolute_sequence_position": len(trace["prompt_token_ids"]) + position,
                "anchor_position": position,
                "prefix_response_tokens": position,
                "student_token_id": int(token_id),
                "student_token_piece": tokenizer.convert_ids_to_tokens(int(token_id)),
                "student_token_text": tokenizer.decode(
                    [int(token_id)], skip_special_tokens=False
                ),
                "student_logprob": student_lp,
                "teacher_logprob_on_student_token": teacher_lp,
                "student_probability": probability(student_lp),
                "teacher_probability_on_student_token": probability(teacher_lp),
                "signed_logprob_gap_student_minus_teacher": signed_gap,
                "absolute_logprob_gap": absolute_gap,
                "gap_direction": (
                    "student_higher" if signed_gap > 0 else "teacher_higher"
                ),
                "threshold": threshold,
                "teacher_top_tokens": top_tokens,
            }
        )
    return candidates


def decoded_prefix(
    trace: dict[str, Any], tokenizer: Any, response_position: int, max_chars: int
) -> tuple[str, bool, int]:
    """Decode s_t, excluding the student token y_t at response_position."""
    token_ids = trace["prompt_token_ids"] + trace["response_token_ids"][:response_position]
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    original_chars = len(text)
    if max_chars > 0 and len(text) > max_chars:
        marker = "[... PREFIX TRUNCATED; SHOWING THE MOST RECENT CONTEXT ...]\n"
        return marker + text[-max_chars:], True, original_chars
    return text, False, original_chars


def selected_anchor_artifact(
    trace: dict[str, Any],
    candidate: dict[str, Any],
    judgment: dict[str, Any],
) -> dict[str, Any]:
    position = int(candidate["response_position"])
    return {
        "status": "ANCHOR_SELECTED",
        **trace_identity(trace),
        "anchor_response_position": position,
        "anchor_absolute_sequence_position": len(trace["prompt_token_ids"]) + position,
        "anchor_prefix_token_ids": (
            trace["prompt_token_ids"] + trace["response_token_ids"][:position]
        ),
        "student_candidate_token_id": candidate["student_token_id"],
        "student_candidate_token_text": candidate["student_token_text"],
        "absolute_logprob_gap": candidate["absolute_logprob_gap"],
        "gap_direction": candidate["gap_direction"],
        "judge": judgment,
    }
