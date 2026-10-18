#!/usr/bin/env python3
"""Select an OCRD anchor from a student trace.

GPU scoring and network judging are separate commands because the cluster GPU
nodes do not have Internet access.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR.parent))

from anchor_selector.core import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    build_candidates,
    decoded_prefix,
    load_trace,
    read_jsonl,
    selected_anchor_artifact,
    trace_identity,
    tokenizer_fingerprint,
)
from anchor_selector.openlux_client import (  # noqa: E402
    OpenLuxClient,
    build_user_prompt,
    load_env_file,
)


DEFAULT_STUDENT = "/storage/yukaichengLab/share/model/Qwen3-1.7B"
DEFAULT_TEACHER = "/storage/yukaichengLab/share/model/Qwen3-14B"


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Config must be a JSON object")
    return value


def selected_trace(args: argparse.Namespace) -> dict[str, Any]:
    return load_trace(
        args.trace,
        row_index=args.row_index,
        dataset_index=args.dataset_index,
        problem_id=args.problem_id,
    )


def score_trace_with_vllm(
    trace: dict[str, Any],
    model_path: str,
    tokenizer_path: str,
    topk: int,
    max_model_len: int,
    gpu_memory_utilization: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, int, str]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    sequence_ids = trace["prompt_token_ids"] + trace["response_token_ids"]
    if len(sequence_ids) + 1 > max_model_len:
        raise ValueError(
            f"Sequence has {len(sequence_ids)} tokens but max_model_len={max_model_len}"
        )
    if max(sequence_ids) >= len(tokenizer):
        raise ValueError("Trace contains token IDs outside tokenizer vocabulary")
    llm = LLM(
        model=model_path,
        tokenizer=tokenizer_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=False,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=1,
        enable_prefix_caching=False,
        disable_log_stats=True,
        max_logprobs=max(1, topk),
    )
    sampling = SamplingParams(
        temperature=1.0,
        max_tokens=1,
        prompt_logprobs=max(1, topk),
        seed=20260822,
    )
    output = llm.generate(
        [TokensPrompt(prompt_token_ids=sequence_ids)], sampling, use_tqdm=False
    )[0]
    prompt_logprobs = output.prompt_logprobs
    if prompt_logprobs is None or len(prompt_logprobs) != len(sequence_ids):
        raise RuntimeError("vLLM did not return aligned prompt logprobs")

    from anchor_selector.core import lookup_actual_logprob, sorted_topk

    response_start = len(trace["prompt_token_ids"])
    chosen: list[float] = []
    top_ids: list[list[int]] = []
    top_logprobs: list[list[float]] = []
    for response_position, token_id in enumerate(trace["response_token_ids"]):
        sequence_position = response_start + response_position
        entry = prompt_logprobs[sequence_position]
        chosen.append(lookup_actual_logprob(entry, int(token_id)))
        if topk:
            ids, logprobs = sorted_topk(entry, topk, len(tokenizer))
            top_ids.append(ids)
            top_logprobs.append(logprobs)
    chosen_array = np.asarray(chosen, dtype=np.float32)
    ids_array = np.asarray(top_ids, dtype=np.int32) if topk else None
    top_lp_array = np.asarray(top_logprobs, dtype=np.float32) if topk else None
    del output, prompt_logprobs, llm
    gc.collect()
    return (
        chosen_array,
        ids_array,
        top_lp_array,
        len(tokenizer),
        tokenizer_fingerprint(tokenizer),
    )


def atomic_npz(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


def command_score_student(args: argparse.Namespace) -> None:
    trace = selected_trace(args)
    chosen, _, _, vocab_size, tokenizer_sha256 = score_trace_with_vllm(
        trace,
        args.student,
        args.tokenizer or args.student,
        topk=0,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    output = args.work_dir / "student_scores.npz"
    atomic_npz(output, chosen_logprobs=chosen)
    atomic_write_json(
        args.work_dir / "student_manifest.json",
        {
            "status": "STUDENT_SCORES_OK",
            **trace_identity(trace),
            "model": args.student,
            "tokenizer": args.tokenizer or args.student,
            "vocab_size": vocab_size,
            "tokenizer_vocab_sha256": tokenizer_sha256,
            "artifact": str(output),
        },
    )
    print(f"STUDENT_SCORES_OK {output}")


def command_score_teacher(args: argparse.Namespace) -> None:
    trace = selected_trace(args)
    config = load_config(args.config)
    topk = int(config["teacher_topk"])
    chosen, top_ids, top_logprobs, vocab_size, tokenizer_sha256 = score_trace_with_vllm(
        trace,
        args.teacher,
        args.tokenizer or args.teacher,
        topk=topk,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    assert top_ids is not None and top_logprobs is not None
    output = args.work_dir / "teacher_scores.npz"
    atomic_npz(
        output,
        chosen_logprobs=chosen,
        topk_ids=top_ids,
        topk_logprobs=top_logprobs,
    )
    atomic_write_json(
        args.work_dir / "teacher_manifest.json",
        {
            "status": "TEACHER_SCORES_OK",
            **trace_identity(trace),
            "model": args.teacher,
            "tokenizer": args.tokenizer or args.teacher,
            "vocab_size": vocab_size,
            "tokenizer_vocab_sha256": tokenizer_sha256,
            "topk": topk,
            "artifact": str(output),
        },
    )
    print(f"TEACHER_SCORES_OK {output}")


def verify_manifests(work_dir: Path, trace: dict[str, Any]) -> None:
    expected = trace_identity(trace)["trace_sha256"]
    for name in ("student_manifest.json", "teacher_manifest.json"):
        path = work_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing scoring artifact: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("trace_sha256") != expected:
            raise ValueError(f"{name} belongs to a different trace")
    student_manifest = json.loads(
        (work_dir / "student_manifest.json").read_text(encoding="utf-8")
    )
    teacher_manifest = json.loads(
        (work_dir / "teacher_manifest.json").read_text(encoding="utf-8")
    )
    student_vocab = student_manifest["vocab_size"]
    teacher_vocab = teacher_manifest["vocab_size"]
    if student_vocab != teacher_vocab:
        raise ValueError(
            f"Student/teacher tokenizer sizes differ: {student_vocab} != {teacher_vocab}"
        )
    if student_manifest.get("tokenizer_vocab_sha256") != teacher_manifest.get(
        "tokenizer_vocab_sha256"
    ):
        raise ValueError("Student/teacher tokenizer token-to-ID maps differ")


def construct_candidates(
    args: argparse.Namespace, trace: dict[str, Any]
) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer

    verify_manifests(args.work_dir, trace)
    config = load_config(args.config)
    threshold = (
        args.gap_threshold
        if args.gap_threshold is not None
        else float(config["gap_threshold"])
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.student,
        local_files_only=True,
        trust_remote_code=False,
    )
    with np.load(args.work_dir / "student_scores.npz") as student:
        student_chosen = student["chosen_logprobs"].copy()
    with np.load(args.work_dir / "teacher_scores.npz") as teacher:
        teacher_chosen = teacher["chosen_logprobs"].copy()
        teacher_ids = teacher["topk_ids"].copy()
        teacher_logprobs = teacher["topk_logprobs"].copy()
    candidates = build_candidates(
        trace,
        tokenizer,
        student_chosen,
        teacher_chosen,
        teacher_ids,
        teacher_logprobs,
        threshold=threshold,
    )
    atomic_write_jsonl(args.work_dir / "candidates.jsonl", candidates)
    atomic_write_json(
        args.work_dir / "candidate_manifest.json",
        {
            "status": "ANCHOR_CANDIDATES_OK",
            **trace_identity(trace),
            "formula": "abs(log_pi_student_y_t - log_pi_teacher_y_t)",
            "strict_comparison": "absolute_gap > threshold",
            "threshold": threshold,
            "candidate_count": len(candidates),
            "positions": [item["response_position"] for item in candidates],
        },
    )
    return candidates


def command_candidates(args: argparse.Namespace) -> None:
    trace = selected_trace(args)
    candidates = construct_candidates(args, trace)
    print(f"ANCHOR_CANDIDATES_OK count={len(candidates)}")


def load_api_key(args: argparse.Namespace) -> str:
    if os.environ.get("OPENLUX_API_KEY"):
        return os.environ["OPENLUX_API_KEY"]
    values = load_env_file(args.secrets)
    key = values.get("OPENLUX_API_KEY", "")
    if not key:
        raise ValueError(f"OPENLUX_API_KEY missing from {args.secrets}")
    return key


def command_judge(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    trace = selected_trace(args)
    candidate_path = args.work_dir / "candidates.jsonl"
    if args.rebuild_candidates or not candidate_path.is_file():
        candidates = construct_candidates(args, trace)
    else:
        candidates = read_jsonl(candidate_path)
    config = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.student,
        local_files_only=True,
        trust_remote_code=False,
    )
    client = OpenLuxClient(
        api_url=str(config["api_url"]),
        api_key=load_api_key(args),
        model=str(config["judge_model"]),
        temperature=float(config["judge_temperature"]),
        max_tokens=int(config["judge_max_tokens"]),
        timeout_seconds=float(config["judge_timeout_seconds"]),
        max_retries=int(config["judge_max_retries"]),
        reasoning_effort=config.get("judge_reasoning_effort"),
    )
    judgments_path = args.work_dir / "judgments.jsonl"
    judgments = read_jsonl(judgments_path) if judgments_path.is_file() else []
    by_position = {int(row["response_position"]): row for row in judgments}
    max_candidates = args.max_candidates if args.max_candidates > 0 else len(candidates)

    selected: tuple[dict[str, Any], dict[str, Any]] | None = None
    for candidate in candidates[:max_candidates]:
        position = int(candidate["response_position"])
        cached = by_position.get(position)
        if cached is not None:
            if cached["judgment"]["select_anchor"]:
                selected = (candidate, cached)
                break
            continue
        prefix_text, truncated, original_chars = decoded_prefix(
            trace,
            tokenizer,
            position,
            int(config["prefix_max_chars"]),
        )
        user_prompt = build_user_prompt(
            trace,
            candidate,
            prefix_text,
            truncated,
            original_chars,
        )
        judgment, api_metadata = client.judge(user_prompt)
        row = {
            "response_position": position,
            "candidate_index": candidate["candidate_index"],
            "absolute_logprob_gap": candidate["absolute_logprob_gap"],
            "judgment": judgment,
            "api": api_metadata,
            "prefix_truncated": truncated,
            "prefix_original_chars": original_chars,
        }
        judgments.append(row)
        judgments.sort(key=lambda item: int(item["response_position"]))
        atomic_write_jsonl(judgments_path, judgments)
        by_position[position] = row
        print(
            f"JUDGED position={position} label={judgment['label']} "
            f"select={judgment['select_anchor']} confidence={judgment['confidence']:.3f}"
        )
        if judgment["select_anchor"]:
            selected = (candidate, row)
            break

    if selected is None:
        artifact = {
            "status": "NO_ANCHOR_SELECTED",
            **trace_identity(trace),
            "candidate_count": len(candidates),
            "judged_count": len(judgments),
            "exhaustive": max_candidates >= len(candidates),
        }
        atomic_write_json(args.work_dir / "selected_anchor.json", artifact)
        print("NO_ANCHOR_SELECTED")
        return
    candidate, row = selected
    artifact = selected_anchor_artifact(trace, candidate, row)
    atomic_write_json(args.work_dir / "selected_anchor.json", artifact)
    print(
        f"ANCHOR_SELECTED response_position={candidate['response_position']} "
        f"output={args.work_dir / 'selected_anchor.json'}"
    )


def command_prepare_judge(args: argparse.Namespace) -> None:
    """Materialize exact prefixes so judging can run on a networked CPU host."""
    from transformers import AutoTokenizer

    trace = selected_trace(args)
    candidate_path = args.work_dir / "candidates.jsonl"
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Missing candidates artifact: {candidate_path}")
    candidates = read_jsonl(candidate_path)
    config = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.student,
        local_files_only=True,
        trust_remote_code=False,
    )
    max_candidates = args.max_candidates if args.max_candidates > 0 else len(candidates)
    prepared: list[dict[str, Any]] = []
    for candidate in candidates[:max_candidates]:
        position = int(candidate["response_position"])
        prefix_text, truncated, original_chars = decoded_prefix(
            trace,
            tokenizer,
            position,
            int(config["prefix_max_chars"]),
        )
        prepared.append(
            {
                "candidate": candidate,
                "prefix_text": prefix_text,
                "prefix_truncated": truncated,
                "prefix_original_chars": original_chars,
            }
        )
    atomic_write_json(args.work_dir / "judge_trace.json", trace)
    atomic_write_jsonl(args.work_dir / "judge_inputs.jsonl", prepared)
    atomic_write_json(
        args.work_dir / "judge_input_manifest.json",
        {
            "status": "JUDGE_INPUTS_OK",
            **trace_identity(trace),
            "candidate_count": len(candidates),
            "prepared_count": len(prepared),
            "prefix_excludes_student_candidate_token": True,
        },
    )
    print(f"JUDGE_INPUTS_OK prepared={len(prepared)}")


def command_judge_prepared(args: argparse.Namespace) -> None:
    """Judge a prepared bundle without loading transformers or model weights."""
    trace_path = args.work_dir / "judge_trace.json"
    inputs_path = args.work_dir / "judge_inputs.jsonl"
    if not trace_path.is_file() or not inputs_path.is_file():
        raise FileNotFoundError(
            f"Missing prepared judge bundle in {args.work_dir}; run prepare-judge first"
        )
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    prepared = read_jsonl(inputs_path)
    config = load_config(args.config)
    client = OpenLuxClient(
        api_url=str(config["api_url"]),
        api_key=load_api_key(args),
        model=str(config["judge_model"]),
        temperature=float(config["judge_temperature"]),
        max_tokens=int(config["judge_max_tokens"]),
        timeout_seconds=float(config["judge_timeout_seconds"]),
        max_retries=int(config["judge_max_retries"]),
        reasoning_effort=config.get("judge_reasoning_effort"),
    )
    judgments_path = args.work_dir / "judgments.jsonl"
    judgments = read_jsonl(judgments_path) if judgments_path.is_file() else []
    by_position = {int(row["response_position"]): row for row in judgments}
    selected: tuple[dict[str, Any], dict[str, Any]] | None = None

    for item in prepared:
        candidate = item["candidate"]
        position = int(candidate["response_position"])
        cached = by_position.get(position)
        if cached is not None:
            if cached["judgment"]["select_anchor"]:
                selected = (candidate, cached)
                break
            continue
        user_prompt = build_user_prompt(
            trace,
            candidate,
            str(item["prefix_text"]),
            bool(item["prefix_truncated"]),
            int(item["prefix_original_chars"]),
        )
        judgment, api_metadata = client.judge(user_prompt)
        row = {
            "response_position": position,
            "candidate_index": candidate["candidate_index"],
            "absolute_logprob_gap": candidate["absolute_logprob_gap"],
            "judgment": judgment,
            "api": api_metadata,
            "prefix_truncated": bool(item["prefix_truncated"]),
            "prefix_original_chars": int(item["prefix_original_chars"]),
        }
        judgments.append(row)
        judgments.sort(key=lambda value: int(value["response_position"]))
        atomic_write_jsonl(judgments_path, judgments)
        by_position[position] = row
        print(
            f"JUDGED position={position} label={judgment['label']} "
            f"select={judgment['select_anchor']} confidence={judgment['confidence']:.3f}",
            flush=True,
        )
        if judgment["select_anchor"]:
            selected = (candidate, row)
            break

    if selected is None:
        artifact = {
            "status": "NO_ANCHOR_SELECTED",
            **trace_identity(trace),
            "candidate_count": int(
                json.loads(
                    (args.work_dir / "candidate_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )["candidate_count"]
            ),
            "judged_count": len(judgments),
            "exhaustive": False,
            "prepared_count": len(prepared),
        }
        atomic_write_json(args.work_dir / "selected_anchor.json", artifact)
        print("NO_ANCHOR_SELECTED", flush=True)
        return
    candidate, row = selected
    artifact = selected_anchor_artifact(trace, candidate, row)
    atomic_write_json(args.work_dir / "selected_anchor.json", artifact)
    print(
        f"ANCHOR_SELECTED response_position={candidate['response_position']} "
        f"output={args.work_dir / 'selected_anchor.json'}",
        flush=True,
    )


def command_probe_api(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    client = OpenLuxClient(
        api_url=str(config["api_url"]),
        api_key=load_api_key(args),
        model=str(config["judge_model"]),
        temperature=0.0,
        max_tokens=120,
        timeout_seconds=float(config["judge_timeout_seconds"]),
        max_retries=0,
        reasoning_effort=config.get("judge_reasoning_effort"),
    )
    prompt = (
        "This is a connectivity test. Return exactly this JSON object: "
        '{"select_anchor": false, "label": "insufficient_evidence", '
        '"confidence": 1.0, "reason": "connectivity test", '
        '"student_direction": "unknown", "teacher_possible_direction": "unknown"}'
    )
    judgment, metadata = client.judge(prompt)
    atomic_write_json(
        args.work_dir / "api_probe.json",
        {"status": "OPENLUX_API_OK", "judgment": judgment, "api": metadata},
    )
    print(
        f"OPENLUX_API_OK request_model={config['judge_model']} "
        f"response_model={metadata.get('response_model')}"
    )


def add_trace_selector(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trace", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--row-index", type=int, default=0)
    selection.add_argument("--dataset-index", type=int)
    selection.add_argument("--problem-id")


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=MODULE_DIR / "config.json")
    parser.add_argument("--tokenizer")
    parser.add_argument("--student", default=DEFAULT_STUDENT)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    student = commands.add_parser("score-student")
    add_trace_selector(student)
    add_common_paths(student)
    student.add_argument("--max-model-len", type=int, default=40960)
    student.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    student.set_defaults(func=command_score_student)

    teacher = commands.add_parser("score-teacher")
    add_trace_selector(teacher)
    add_common_paths(teacher)
    teacher.add_argument("--teacher", default=DEFAULT_TEACHER)
    teacher.add_argument("--max-model-len", type=int, default=40960)
    teacher.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    teacher.set_defaults(func=command_score_teacher)

    candidates = commands.add_parser("candidates")
    add_trace_selector(candidates)
    add_common_paths(candidates)
    candidates.add_argument("--gap-threshold", type=float)
    candidates.set_defaults(func=command_candidates)

    judge = commands.add_parser("judge")
    add_trace_selector(judge)
    add_common_paths(judge)
    judge.add_argument("--gap-threshold", type=float)
    judge.add_argument("--secrets", type=Path, default=MODULE_DIR / ".secrets.env")
    judge.add_argument("--max-candidates", type=int, default=0)
    judge.add_argument("--rebuild-candidates", action="store_true")
    judge.set_defaults(func=command_judge)

    prepare = commands.add_parser("prepare-judge")
    add_trace_selector(prepare)
    add_common_paths(prepare)
    prepare.add_argument("--max-candidates", type=int, default=0)
    prepare.set_defaults(func=command_prepare_judge)

    prepared = commands.add_parser("judge-prepared")
    prepared.add_argument("--work-dir", type=Path, required=True)
    prepared.add_argument("--config", type=Path, default=MODULE_DIR / "config.json")
    prepared.add_argument("--secrets", type=Path, default=MODULE_DIR / ".secrets.env")
    prepared.set_defaults(func=command_judge_prepared)

    probe = commands.add_parser("probe-api")
    probe.add_argument("--work-dir", type=Path, required=True)
    probe.add_argument("--config", type=Path, default=MODULE_DIR / "config.json")
    probe.add_argument("--secrets", type=Path, default=MODULE_DIR / ".secrets.env")
    probe.set_defaults(func=command_probe_api)
    return root


def main() -> None:
    args = parser().parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
