"""Pure validation and artifact assembly for teacher bridges."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_selected_anchor(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Anchor artifact must be a JSON object")
    if value.get("status") != "ANCHOR_SELECTED":
        raise ValueError(f"Anchor is not selected: status={value.get('status')!r}")
    prefix = value.get("anchor_prefix_token_ids")
    if not isinstance(prefix, list) or not prefix:
        raise ValueError("anchor_prefix_token_ids must be a non-empty list")
    if not all(isinstance(token_id, int) and token_id >= 0 for token_id in prefix):
        raise TypeError("anchor_prefix_token_ids must contain non-negative integers")
    student_token = value.get("student_candidate_token_id")
    if not isinstance(student_token, int) or student_token < 0:
        raise ValueError("student_candidate_token_id must be a non-negative integer")
    return value


def validate_bridge_tokens(
    bridge_token_ids: list[int], student_token_id: int, requested_length: int
) -> None:
    if requested_length < 1:
        raise ValueError("Bridge length h must be at least 1")
    if len(bridge_token_ids) != requested_length:
        raise ValueError(
            f"Bridge has {len(bridge_token_ids)} tokens; expected {requested_length}"
        )
    if bridge_token_ids[0] == student_token_id:
        raise ValueError("Bridge first token equals the student anchor token")
    if not all(isinstance(token_id, int) and token_id >= 0 for token_id in bridge_token_ids):
        raise TypeError("Bridge token IDs must be non-negative integers")


def assemble_bridge_artifact(
    *,
    anchor: dict[str, Any],
    anchor_path: Path,
    teacher_model: str,
    tokenizer_path: str,
    requested_length: int,
    bridge_token_ids: list[int],
    bridge_token_texts: list[str],
    bridge_text: str,
    chosen_logprobs: list[float | None],
    sampling: dict[str, Any],
    first_step: dict[str, Any],
    continuation_finish_reason: str | None,
) -> dict[str, Any]:
    student_token_id = int(anchor["student_candidate_token_id"])
    validate_bridge_tokens(bridge_token_ids, student_token_id, requested_length)
    if len(bridge_token_texts) != requested_length:
        raise ValueError("bridge_token_texts is not aligned with bridge_token_ids")
    if len(chosen_logprobs) != requested_length:
        raise ValueError("chosen_logprobs is not aligned with bridge_token_ids")
    prefix = [int(token_id) for token_id in anchor["anchor_prefix_token_ids"]]
    return {
        "status": "BRIDGE_GENERATED",
        "schema_version": 1,
        "dataset_index": anchor.get("dataset_index"),
        "problem_id": anchor.get("problem_id"),
        "trace_sha256": anchor.get("trace_sha256"),
        "anchor_artifact": str(anchor_path),
        "anchor_response_position": anchor.get("anchor_response_position"),
        "anchor_absolute_sequence_position": anchor.get(
            "anchor_absolute_sequence_position"
        ),
        "anchor_prefix_token_count": len(prefix),
        "student_anchor_token_id": student_token_id,
        "student_anchor_token_text": anchor.get("student_candidate_token_text"),
        "teacher_model": teacher_model,
        "tokenizer": tokenizer_path,
        "requested_bridge_length": requested_length,
        "actual_bridge_length": len(bridge_token_ids),
        "bridge_token_ids": bridge_token_ids,
        "bridge_token_texts": bridge_token_texts,
        "bridge_chosen_logprobs": chosen_logprobs,
        "bridge_text": bridge_text,
        "bridge_first_token_id": bridge_token_ids[0],
        "bridge_first_token_text": bridge_token_texts[0],
        "first_token_differs_from_student": bridge_token_ids[0]
        != student_token_id,
        "first_token_constraint": {
            "method": "direct_single_token_id_bad_words_on_first_decode_step",
            "excluded_token_ids": [student_token_id],
            "constraint_applies_only_to_first_bridge_token": True,
        },
        "first_step": first_step,
        "sampling": sampling,
        "continuation_finish_reason": continuation_finish_reason,
        "bridge_prefix_token_ids": prefix + bridge_token_ids,
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)
