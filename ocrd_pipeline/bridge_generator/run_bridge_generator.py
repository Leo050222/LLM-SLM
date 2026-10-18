#!/usr/bin/env python3
"""Generate an exact-length teacher bridge from a selected anchor."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR.parent))

from bridge_generator.core import (  # noqa: E402
    assemble_bridge_artifact,
    atomic_write_json,
    load_selected_anchor,
)


DEFAULT_TEACHER = "/storage/yukaichengLab/share/model/Qwen3-14B"


def logprob_value(value: Any) -> float:
    if hasattr(value, "logprob"):
        return float(value.logprob)
    if isinstance(value, dict) and "logprob" in value:
        return float(value["logprob"])
    return float(value)


def chosen_logprob(entry: Any, token_id: int) -> float | None:
    if entry is None:
        return None
    value = entry.get(token_id)
    if value is None:
        value = entry.get(str(token_id))
    if value is None:
        return None
    result = logprob_value(value)
    return result if math.isfinite(result) else None


def serialize_top_logprobs(entry: Any, tokenizer: Any, limit: int) -> list[dict[str, Any]]:
    if entry is None:
        return []
    rows: list[tuple[float, int]] = []
    for raw_token_id, raw_value in entry.items():
        token_id = int(raw_token_id)
        value = logprob_value(raw_value)
        if math.isfinite(value):
            rows.append((value, token_id))
    rows.sort(key=lambda item: (-item[0], item[1]))
    return [
        {
            "rank": rank,
            "token_id": token_id,
            "token_text": tokenizer.decode([token_id], skip_special_tokens=False),
            "logprob": value,
            "probability": math.exp(value) if value > -745.0 else 0.0,
        }
        for rank, (value, token_id) in enumerate(rows[:limit], start=1)
    ]


def bad_word_token_sequences(tokenizer: Any, bad_word: str) -> list[list[int]]:
    """Mirror vLLM's leading-space bad-word tokenization for preflight."""
    sequences: list[list[int]] = []
    for add_prefix_space in (False, True):
        prefix = " " if add_prefix_space else ""
        prompt = prefix + bad_word.lstrip()
        token_ids = [
            int(token_id)
            for token_id in tokenizer.encode(prompt, add_special_tokens=False)
        ]
        if not token_ids:
            continue
        if (not add_prefix_space) or (
            token_ids[0] != sequences[-1][0]
            and len(token_ids) == len(sequences[-1])
        ):
            sequences.append(token_ids)
    return sequences


def generate_bridge(args: argparse.Namespace) -> dict[str, Any]:
    if args.h < 1:
        raise ValueError("--h must be at least 1")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    anchor = load_selected_anchor(args.anchor)
    prefix_ids = [int(token_id) for token_id in anchor["anchor_prefix_token_ids"]]
    student_token_id = int(anchor["student_candidate_token_id"])
    tokenizer_path = args.tokenizer or args.teacher
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    vocab_size = len(tokenizer)
    if not 0 <= student_token_id < vocab_size:
        raise ValueError(
            f"Student anchor token {student_token_id} outside tokenizer vocabulary"
        )
    if len(prefix_ids) + args.h > args.max_model_len:
        raise ValueError(
            f"prefix ({len(prefix_ids)}) + bridge ({args.h}) exceeds "
            f"max_model_len={args.max_model_len}"
        )

    llm = LLM(
        model=args.teacher,
        tokenizer=tokenizer_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        enable_prefix_caching=False,
        disable_log_stats=True,
        max_logprobs=max(1, args.first_step_logprobs),
    )

    # Hard masking is used only for the first decode step. A decoded token
    # does not always encode back to the same single token, so populate
    # vLLM's already-tokenized bad-word field directly instead of using the
    # text-valued bad_words argument.
    student_token_text = tokenizer.decode(
        [student_token_id], skip_special_tokens=False
    )
    first_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        max_tokens=1,
        min_tokens=1,
        ignore_eos=True,
        logprobs=args.first_step_logprobs,
        _bad_words_token_ids=[[student_token_id]],
        detokenize=False,
        skip_special_tokens=False,
    )
    bad_token_sequences = [[student_token_id]]
    first_request = llm.generate(
        [TokensPrompt(prompt_token_ids=prefix_ids)], first_params, use_tqdm=False
    )[0]
    first_output = first_request.outputs[0]
    first_ids = [int(token_id) for token_id in first_output.token_ids]
    if len(first_ids) != 1:
        raise RuntimeError(f"First decode produced {len(first_ids)} tokens; expected 1")
    first_id = first_ids[0]
    if first_id == student_token_id:
        raise RuntimeError("Hard first-token exclusion constraint was violated")
    first_entry = first_output.logprobs[0] if first_output.logprobs else None
    bridge_ids = [first_id]
    bridge_logprobs: list[float | None] = [chosen_logprob(first_entry, first_id)]
    continuation_finish_reason: str | None = None

    if args.h > 1:
        remaining = args.h - 1
        continuation_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed + 1,
            max_tokens=remaining,
            min_tokens=remaining,
            ignore_eos=True,
            logprobs=1,
            detokenize=False,
            skip_special_tokens=False,
        )
        continuation_request = llm.generate(
            [TokensPrompt(prompt_token_ids=prefix_ids + [first_id])],
            continuation_params,
            use_tqdm=False,
        )[0]
        continuation_output = continuation_request.outputs[0]
        continuation_ids = [int(token_id) for token_id in continuation_output.token_ids]
        if len(continuation_ids) != remaining:
            raise RuntimeError(
                f"Continuation produced {len(continuation_ids)} tokens; expected {remaining}"
            )
        bridge_ids.extend(continuation_ids)
        continuation_finish_reason = continuation_output.finish_reason
        entries = continuation_output.logprobs or [None] * remaining
        bridge_logprobs.extend(
            chosen_logprob(entry, token_id)
            for entry, token_id in zip(entries, continuation_ids, strict=True)
        )

    token_texts = [
        tokenizer.decode([token_id], skip_special_tokens=False)
        for token_id in bridge_ids
    ]
    artifact = assemble_bridge_artifact(
        anchor=anchor,
        anchor_path=args.anchor,
        teacher_model=args.teacher,
        tokenizer_path=tokenizer_path,
        requested_length=args.h,
        bridge_token_ids=bridge_ids,
        bridge_token_texts=token_texts,
        bridge_text=tokenizer.decode(bridge_ids, skip_special_tokens=False),
        chosen_logprobs=bridge_logprobs,
        sampling={
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "first_step_seed": args.seed,
            "continuation_seed": args.seed + 1 if args.h > 1 else None,
            "ignore_eos": True,
            "exact_length_enforced_by_min_tokens_equals_max_tokens": True,
        },
        first_step={
            "finish_reason": first_output.finish_reason,
            "chosen_token_id": first_id,
            "chosen_token_text": token_texts[0],
            "chosen_logprob": bridge_logprobs[0],
            "top_logprobs_after_first_token_constraint": serialize_top_logprobs(
                first_entry, tokenizer, args.first_step_logprobs
            ),
            "bad_word_token_sequences": bad_token_sequences,
        },
        continuation_finish_reason=continuation_finish_reason,
    )
    return artifact


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--anchor", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--h", type=int, required=True)
    result.add_argument("--teacher", default=DEFAULT_TEACHER)
    result.add_argument("--tokenizer")
    result.add_argument("--temperature", type=float, default=1.0)
    result.add_argument("--top-p", type=float, default=1.0)
    result.add_argument("--top-k", type=int, default=0)
    result.add_argument("--seed", type=int, default=20260822)
    result.add_argument("--first-step-logprobs", type=int, default=16)
    result.add_argument("--max-model-len", type=int, default=40960)
    result.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    return result


def main() -> None:
    args = parser().parse_args()
    artifact = generate_bridge(args)
    atomic_write_json(args.output, artifact)
    print(
        "BRIDGE_GENERATED "
        f"h={artifact['actual_bridge_length']} "
        f"first_token_id={artifact['bridge_first_token_id']} "
        f"differs={artifact['first_token_differs_from_student']} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
