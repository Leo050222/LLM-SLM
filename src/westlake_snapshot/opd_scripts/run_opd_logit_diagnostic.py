#!/usr/bin/env python3
"""Frozen step-0 token-level OPD diagnostics for Qwen3 student rollouts.

The pipeline intentionally separates generation, teacher scoring, and student
scoring into distinct processes so every phase releases GPU memory completely.
No optimizer is constructed and no backward pass is performed.
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_STUDENT = "/storage/yukaichengLab/share/model/Qwen3-4B"
DEFAULT_TEACHER = "/storage/yukaichengLab/share/model/Qwen3-14B"
DEFAULT_DATA = (
    "/storage/yukaichengLab/wangxin/test_eiloeo/"
    "opd_eval_hard/modelscope_cache/evalscope/aime24/"
    "test-00000-of-00001.parquet"
)
TOPK = 128
SEED = 20260806
MAX_TOKENS = 38912
MAX_MODEL_LEN = 40960


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_dataset(path: Path) -> pd.DataFrame:
    """Load the local EvalScope benchmark formats used by this diagnostic."""
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset format: {path}")


def boxed_expressions(text: str) -> list[str]:
    r"""Return balanced-brace contents of every LaTeX \boxed expression."""
    results: list[str] = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        depth = 1
        cursor = match.end()
        start = cursor
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            results.append(text[start : cursor - 1])
    return results


def normalize_aime_answer(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().replace(",", "").replace(" ", "")
    normalized = normalized.replace("\\,", "")
    if re.fullmatch(r"[+-]?\d+", normalized):
        return str(int(normalized))
    return normalized


def score_aime_response(response: str, solution: str) -> tuple[bool, str | None, str | None]:
    # EvalScope removes reasoning up to </think> before strict answer scoring.
    final_region = response.rsplit("</think>", 1)[-1]
    predicted_boxes = boxed_expressions(final_region)
    target_boxes = boxed_expressions(solution)
    predicted = normalize_aime_answer(predicted_boxes[-1] if predicted_boxes else None)
    target = normalize_aime_answer(target_boxes[-1] if target_boxes else solution)
    return predicted is not None and predicted == target, predicted, target


def aime_user_prompt(problem: str) -> str:
    return (
        "\nSolve the following math problem step by step. Put your answer inside "
        "\\boxed{}.\n\n"
        f"{problem}\n\n"
        "Remember to put your answer inside \\boxed{}."
    )


def deepmath_user_prompt(problem: str) -> str:
    return (
        "Solve the following math problem. Show your reasoning and put your "
        "final answer inside \\boxed{}.\n\n"
        f"{problem}"
    )


def _math_equiv_worker(candidate: str, target: str, connection: Any) -> None:
    try:
        from verl.utils.reward_score.math_reward import is_equiv

        connection.send(bool(is_equiv(candidate, target)))
    except BaseException:
        connection.send(False)
    finally:
        connection.close()


def score_math_response(
    response: str, target: str, timeout_seconds: float = 20.0
) -> tuple[bool, str | None, str]:
    """Score the declared final answer, with symbolic work in a subprocess."""
    from verl.utils.reward_score.math_reward import strip_string

    final_region = response.rsplit("</think>", 1)[-1].strip()
    boxes = boxed_expressions(final_region)
    candidate = boxes[-1].strip() if boxes else None
    if candidate is None:
        tail = final_region[-1200:]
        markers = list(
            re.finditer(
                r"(?is)(?:final\s+answer|answer\s+is|answer\s*:|therefore[, ]+the\s+answer\s+is|thus[, ]+the\s+answer\s+is)\s*[:=]?\s*",
                tail,
            )
        )
        if markers:
            segment = tail[markers[-1].end() :].strip()
            math_spans = re.findall(r"\$\$([^$]+)\$\$|\$([^$]+)\$", segment, flags=re.S)
            if math_spans:
                candidate = next(value for value in reversed(math_spans[-1]) if value)
            else:
                lines = [line.strip() for line in segment.splitlines() if line.strip()]
                candidate = lines[-1] if lines else None
    if candidate is None:
        return False, None, "missing"

    candidate = candidate.strip().rstrip(".* ")
    try:
        if strip_string(candidate) == strip_string(str(target)):
            return True, candidate, "normalized_exact"
    except Exception:
        pass

    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_math_equiv_worker, args=(candidate, str(target), child), daemon=True
    )
    process.start()
    child.close()
    try:
        if parent.poll(timeout_seconds):
            return bool(parent.recv()), candidate, "symbolic"
        return False, candidate, "timeout"
    except (EOFError, OSError):
        return False, candidate, "error"
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)


def phase_generate(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = read_dataset(args.data)
    if args.limit is not None:
        frame = frame.iloc[: args.limit]
    tokenizer_path = args.tokenizer or args.student
    thinking_mode = not args.disable_thinking
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    prompts: list[TokensPrompt] = []
    metadata: list[dict[str, Any]] = []
    for local_index, row in frame.reset_index(drop=True).iterrows():
        dataset_index = int(row.get("dataset_index", local_index))
        prompt_builder = aime_user_prompt if args.prompt_style == "aime" else deepmath_user_prompt
        message = prompt_builder(str(row["problem"]))
        encoded_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": message}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=thinking_mode,
        )
        # Transformers 5.6 returns BatchEncoding here even when return_dict is
        # not explicitly requested; older releases returned list[int].
        prompt_ids = (
            encoded_prompt["input_ids"]
            if isinstance(encoded_prompt, Mapping)
            else encoded_prompt
        )
        if not prompt_ids or not all(isinstance(token_id, int) for token_id in prompt_ids):
            raise TypeError(f"Expected integer prompt token IDs, got {type(encoded_prompt)!r}")
        prompts.append(TokensPrompt(prompt_token_ids=prompt_ids))
        problem_id = row.get("id", row.get("problem_idx", dataset_index))
        solution = row.get("solution", row.get("answer"))
        if solution is None:
            raise KeyError(f"Dataset row {dataset_index} has neither solution nor answer")
        metadata.append(
            {
                "dataset": args.dataset_name,
                "dataset_index": int(dataset_index),
                "problem_id": str(problem_id),
                "problem": str(row["problem"]),
                "solution": str(solution),
                "prompt_text": message,
                "prompt_token_ids": prompt_ids,
                "difficulty": float(row["difficulty"]) if "difficulty" in row else None,
                "topic": str(row["topic"]) if "topic" in row else None,
                "source_index": int(row["source_index"]) if "source_index" in row else None,
            }
        )

    engine = LLM(
        model=args.student,
        tokenizer=tokenizer_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=False,
        seed=SEED,
        gpu_memory_utilization=0.90,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=min(8, max(1, len(prompts))),
        enable_prefix_caching=True,
        disable_log_stats=True,
    )
    sampling = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=MAX_TOKENS,
        n=1,
        seed=SEED,
    )
    started = time.time()
    outputs = engine.generate(prompts, sampling, use_tqdm=True)
    rows: list[dict[str, Any]] = []
    for item, request_output in zip(metadata, outputs, strict=True):
        generated = request_output.outputs[0]
        response_ids = list(generated.token_ids)
        response = tokenizer.decode(response_ids, skip_special_tokens=False)
        if args.scorer == "aime":
            correct, predicted, target = score_aime_response(response, item["solution"])
            score_method = "aime_exact"
        else:
            correct, predicted, score_method = score_math_response(
                response, item["solution"], args.score_timeout
            )
            target = item["solution"]
        rows.append(
            {
                **item,
                "response_token_ids": response_ids,
                "response": response,
                "response_tokens": len(response_ids),
                "finish_reason": generated.finish_reason,
                "stop_reason": generated.stop_reason,
                "correct": bool(correct),
                "predicted_answer": predicted,
                "target_answer": target,
                "score_method": score_method,
            }
        )
    write_jsonl(output_dir / "rollouts.jsonl", rows)
    write_json(
        output_dir / "run_config.json",
        {
            "dataset": args.dataset_name,
            "data_path": str(args.data),
            "student": args.student,
            "tokenizer": tokenizer_path,
            "teacher": args.teacher,
            "student_vocab_size": len(tokenizer),
            "num_prompts": len(rows),
            "sampling": {
                "max_tokens": MAX_TOKENS,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "do_sample": True,
                "n": 1,
                "seed": SEED,
            },
            "topk": TOPK,
            "max_model_len": MAX_MODEL_LEN,
            "prompt_style": args.prompt_style,
            "scorer": args.scorer,
            "thinking_mode": thinking_mode,
            "elapsed_seconds_generate": time.time() - started,
            "parameters_frozen": True,
            "optimizer_created": False,
            "backward_called": False,
        },
    )


def _topk_prompt_arrays(request_output: Any, sequence_ids: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract teacher top-k rows aligned to causal-logit positions."""
    from verl.workers.rollout.vllm_rollout.utils import extract_prompt_logprobs

    result: dict[str, list] = {}
    extract_prompt_logprobs(request_output, TOPK, result)
    topk_ids = np.asarray(result["prompt_ids"], dtype=np.int32)
    topk_logprobs = np.asarray(result["prompt_logprobs"], dtype=np.float32)
    chosen = np.full(len(sequence_ids), np.nan, dtype=np.float32)
    # output.prompt_logprobs[j] is the distribution predicting sequence_ids[j].
    for position in range(1, len(sequence_ids)):
        entry = request_output.prompt_logprobs[position]
        token_id = sequence_ids[position]
        token_data = entry.get(token_id)
        if token_data is None:
            token_data = entry.get(str(token_id))
        if token_data is None:
            raise KeyError(f"Actual token {token_id} absent at prompt position {position}")
        chosen[position - 1] = float(token_data.logprob)
    return topk_ids, topk_logprobs, chosen


def phase_teacher(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    output_dir: Path = args.output_dir
    score_dir = output_dir / "teacher_topk"
    score_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(output_dir / "rollouts.jsonl")
    student_tokenizer = AutoTokenizer.from_pretrained(
        args.student, local_files_only=True, trust_remote_code=False
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        args.teacher, local_files_only=True, trust_remote_code=False
    )
    if student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("Student and teacher tokenizer vocabularies differ")
    engine = LLM(
        model=args.teacher,
        tokenizer=args.teacher,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=False,
        seed=SEED,
        gpu_memory_utilization=0.90,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=1,
        enable_prefix_caching=False,
        disable_log_stats=True,
        max_logprobs=TOPK,
    )
    sampling = SamplingParams(
        temperature=1.0,
        max_tokens=1,
        prompt_logprobs=TOPK,
        seed=SEED,
    )
    started = time.time()
    manifest: list[dict[str, Any]] = []
    for row in rows:
        shard = score_dir / f"example_{row['dataset_index']:03d}.npz"
        sequence_ids = row["prompt_token_ids"] + row["response_token_ids"]
        outputs = engine.generate(
            [TokensPrompt(prompt_token_ids=sequence_ids)], sampling, use_tqdm=False
        )
        all_ids, all_logprobs, all_chosen = _topk_prompt_arrays(outputs[0], sequence_ids)
        start = len(row["prompt_token_ids"]) - 1
        stop = start + len(row["response_token_ids"])
        teacher_ids = all_ids[start:stop]
        teacher_logprobs = all_logprobs[start:stop]
        teacher_chosen = all_chosen[start:stop]
        expected = (len(row["response_token_ids"]), TOPK)
        if teacher_ids.shape != expected or teacher_logprobs.shape != expected:
            raise ValueError(
                f"Teacher shape mismatch for {row['dataset_index']}: "
                f"{teacher_ids.shape}, {teacher_logprobs.shape}, expected {expected}"
            )
        np.savez_compressed(
            shard,
            teacher_topk_ids=teacher_ids,
            teacher_topk_logprobs=teacher_logprobs,
            teacher_chosen_logprob=teacher_chosen,
        )
        manifest.append(
            {
                "dataset_index": row["dataset_index"],
                "response_tokens": len(row["response_token_ids"]),
                "path": str(shard),
                "teacher_mass_mean": float(np.exp(teacher_logprobs).sum(axis=1).mean()),
            }
        )
        del outputs, all_ids, all_logprobs, all_chosen
        gc.collect()
    write_json(output_dir / "teacher_manifest.json", manifest)
    config = json.loads((output_dir / "run_config.json").read_text())
    config["teacher_vocab_size"] = len(teacher_tokenizer)
    config["tokenizers_equal"] = True
    config["elapsed_seconds_teacher"] = time.time() - started
    write_json(output_dir / "run_config.json", config)


def _quantiles(values: pd.Series) -> dict[str, float]:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return {}
    levels = [0.0, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0]
    names = ["min", "p50", "p75", "p90", "p95", "p99", "p99_9", "max"]
    return {name: float(value) for name, value in zip(names, np.quantile(clean, levels), strict=True)}


def _decode_top(tokenizer: Any, ids: np.ndarray, logprobs: np.ndarray, count: int = 10) -> list[dict[str, Any]]:
    return [
        {
            "token_id": int(token_id),
            "token": tokenizer.decode([int(token_id)], skip_special_tokens=False),
            "logprob": float(logprob),
        }
        for token_id, logprob in zip(ids[:count], logprobs[:count], strict=True)
    ]


def phase_student(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir: Path = args.output_dir
    token_dir = output_dir / "token_shards"
    token_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(output_dir / "rollouts.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(
        args.student, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.student,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    started = time.time()
    extreme_rows: list[dict[str, Any]] = []

    with torch.inference_mode():
        for row in rows:
            sequence_ids = row["prompt_token_ids"] + row["response_token_ids"]
            input_ids = torch.tensor(sequence_ids, dtype=torch.long, device="cuda").unsqueeze(0)
            base_output = model.model(input_ids=input_ids, use_cache=False, return_dict=True)
            hidden = base_output.last_hidden_state[0]
            del base_output, input_ids

            teacher_file = output_dir / "teacher_topk" / f"example_{row['dataset_index']:03d}.npz"
            teacher = np.load(teacher_file)
            teacher_ids_np = teacher["teacher_topk_ids"][:, :TOPK]
            teacher_lp_np = teacher["teacher_topk_logprobs"][:, :TOPK]
            teacher_chosen_np = teacher["teacher_chosen_logprob"]
            if teacher_ids_np.shape[1] != TOPK or teacher_lp_np.shape[1] != TOPK:
                raise ValueError(
                    f"Teacher cache has fewer than requested top-k={TOPK} entries: "
                    f"{teacher_ids_np.shape}, {teacher_lp_np.shape}"
                )
            response_ids_np = np.asarray(row["response_token_ids"], dtype=np.int32)
            response_length = len(response_ids_np)
            prediction_start = len(row["prompt_token_ids"]) - 1

            collected: dict[str, list[np.ndarray]] = {
                "student_chosen_logprob": [],
                "teacher_chosen_logprob": [],
                "chosen_logprob_gap": [],
                "abs_chosen_gap": [],
                "sampled_reverse_kl_k1": [],
                "sampled_reverse_kl_k3": [],
                "sampled_reverse_kl_k3_clipped20": [],
                "raw_forward_kl_top128": [],
                "effective_forward_kl_top128": [],
                "teacher_mass_top128": [],
                "student_mass_on_teacher_top128": [],
                "topk_overlap": [],
                "top1_agreement": [],
                "student_entropy": [],
                "teacher_entropy_lower_bound": [],
            }
            example_candidates: list[dict[str, Any]] = []
            chunk_size = 256
            for chunk_start in range(0, response_length, chunk_size):
                chunk_stop = min(response_length, chunk_start + chunk_size)
                positions = torch.arange(
                    prediction_start + chunk_start,
                    prediction_start + chunk_stop,
                    device="cuda",
                )
                logits = model.lm_head(hidden.index_select(0, positions)).float()
                log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
                teacher_ids = torch.from_numpy(
                    teacher_ids_np[chunk_start:chunk_stop]
                ).to(device="cuda", dtype=torch.long)
                teacher_lp = torch.from_numpy(
                    teacher_lp_np[chunk_start:chunk_stop]
                ).to(device="cuda", dtype=torch.float32)
                chosen_ids = torch.from_numpy(
                    response_ids_np[chunk_start:chunk_stop]
                ).to(device="cuda", dtype=torch.long)

                student_at_teacher = logits.gather(1, teacher_ids) - log_z
                student_chosen = logits.gather(1, chosen_ids[:, None]).squeeze(1) - log_z.squeeze(1)
                teacher_chosen = torch.from_numpy(
                    teacher_chosen_np[chunk_start:chunk_stop]
                ).to(device="cuda", dtype=torch.float32)
                teacher_prob = teacher_lp.exp()
                raw_kl = (teacher_prob * (teacher_lp - student_at_teacher)).sum(dim=-1)
                effective_student = student_at_teacher.clamp_min(-10.0)
                effective_teacher = teacher_lp.clamp_min(-10.0)
                effective_kl = (
                    effective_teacher.exp() * (effective_teacher - effective_student)
                ).sum(dim=-1).clamp(-10.0, 10.0)
                student_top_lp, student_top_ids = torch.topk(
                    logits - log_z, k=TOPK, dim=-1
                )
                overlap = (
                    teacher_ids.unsqueeze(-1) == student_top_ids.unsqueeze(-2)
                ).any(dim=-1).sum(dim=-1).float() / TOPK
                probabilities = torch.softmax(logits, dim=-1)
                student_entropy = -(probabilities * (logits - log_z)).sum(dim=-1)
                teacher_mass = teacher_prob.sum(dim=-1)
                teacher_tail = (1.0 - teacher_mass).clamp_min(0.0)
                teacher_entropy_lb = -(teacher_prob * teacher_lp).sum(dim=-1)
                teacher_entropy_lb -= torch.where(
                    teacher_tail > 0,
                    teacher_tail * teacher_tail.clamp_min(1e-30).log(),
                    torch.zeros_like(teacher_tail),
                )

                chosen_gap = teacher_chosen - student_chosen
                values = {
                    "student_chosen_logprob": student_chosen,
                    "teacher_chosen_logprob": teacher_chosen,
                    "chosen_logprob_gap": chosen_gap,
                    "abs_chosen_gap": chosen_gap.abs(),
                    # With actions sampled from the student, E[k1] and E[k3]
                    # estimate KL(student || teacher), i.e. reverse KL.
                    "sampled_reverse_kl_k1": -chosen_gap,
                    "sampled_reverse_kl_k3": torch.expm1(chosen_gap.clamp(max=80.0))
                    - chosen_gap,
                    "sampled_reverse_kl_k3_clipped20": torch.expm1(
                        chosen_gap.clamp(-20.0, 20.0)
                    )
                    - chosen_gap.clamp(-20.0, 20.0),
                    "raw_forward_kl_top128": raw_kl,
                    "effective_forward_kl_top128": effective_kl,
                    "teacher_mass_top128": teacher_mass,
                    "student_mass_on_teacher_top128": student_at_teacher.exp().sum(dim=-1),
                    "topk_overlap": overlap,
                    "top1_agreement": (teacher_ids[:, 0] == student_top_ids[:, 0]).float(),
                    "student_entropy": student_entropy,
                    "teacher_entropy_lower_bound": teacher_entropy_lb,
                }
                for name, tensor in values.items():
                    collected[name].append(tensor.detach().cpu().numpy())

                # Retain rich top-k details only for the strongest tokens in each chunk.
                local_count = min(4, chunk_stop - chunk_start)
                local_indices = torch.topk(raw_kl, k=local_count).indices.cpu().tolist()
                for local_index in local_indices:
                    absolute_response_index = chunk_start + local_index
                    example_candidates.append(
                        {
                            "dataset_index": row["dataset_index"],
                            "problem_id": row["problem_id"],
                            "response_position": absolute_response_index,
                            "token_id": int(response_ids_np[absolute_response_index]),
                            "token": tokenizer.decode(
                                [int(response_ids_np[absolute_response_index])],
                                skip_special_tokens=False,
                            ),
                            "raw_forward_kl_top128": float(raw_kl[local_index].item()),
                            "teacher_top10": _decode_top(
                                tokenizer,
                                teacher_ids_np[absolute_response_index],
                                teacher_lp_np[absolute_response_index],
                            ),
                            "student_top10": _decode_top(
                                tokenizer,
                                student_top_ids[local_index].cpu().numpy(),
                                student_top_lp[local_index].cpu().numpy(),
                            ),
                        }
                    )
                del logits, log_z, probabilities, student_top_lp, student_top_ids

            flat = {name: np.concatenate(parts) for name, parts in collected.items()}
            pieces = tokenizer.convert_ids_to_tokens(response_ids_np.tolist())
            decoded = [
                tokenizer.decode([int(token_id)], skip_special_tokens=False)
                for token_id in response_ids_np
            ]
            think_end_positions = np.flatnonzero(response_ids_np == think_end_id)
            think_end = int(think_end_positions[0]) if think_end_positions.size else response_length
            shard_frame = pd.DataFrame(
                {
                    "dataset": row["dataset"],
                    "dataset_index": row["dataset_index"],
                    "problem_id": row["problem_id"],
                    "correct": row["correct"],
                    "finish_reason": row["finish_reason"],
                    "difficulty": row.get("difficulty"),
                    "topic": row.get("topic"),
                    "response_length": response_length,
                    "response_position": np.arange(response_length, dtype=np.int32),
                    "relative_position": np.arange(response_length, dtype=np.float32)
                    / max(response_length - 1, 1),
                    "token_id": response_ids_np,
                    "token_piece": pieces,
                    "token_text": decoded,
                    "region": np.where(
                        np.arange(response_length) <= think_end, "thinking", "final"
                    ),
                    **flat,
                }
            )
            shard = token_dir / f"example_{row['dataset_index']:03d}.parquet"
            shard_frame.to_parquet(shard, index=False, compression="zstd")
            example_candidates.sort(
                key=lambda item: item["raw_forward_kl_top128"], reverse=True
            )
            extreme_rows.extend(example_candidates[:20])
            del hidden, teacher, shard_frame, flat, collected
            gc.collect()
            torch.cuda.empty_cache()

    extreme_rows.sort(key=lambda item: item["raw_forward_kl_top128"], reverse=True)
    write_jsonl(output_dir / "extreme_tokens.jsonl", extreme_rows)
    config = json.loads((output_dir / "run_config.json").read_text())
    config["diagnostic_topk"] = TOPK
    config["elapsed_seconds_student"] = time.time() - started
    config["student_parameters_require_grad"] = any(
        parameter.requires_grad for parameter in model.parameters()
    )
    write_json(output_dir / "run_config.json", config)


def _position_summary(frame: pd.DataFrame, bin_column: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bin_value, group in frame.groupby(bin_column, observed=True):
        rows.append(
            {
                "bin": int(bin_value),
                "tokens": int(len(group)),
                "active_rollouts": int(group["dataset_index"].nunique()),
                "raw_kl_p50": float(group["raw_forward_kl_top128"].median()),
                "raw_kl_p90": float(group["raw_forward_kl_top128"].quantile(0.9)),
                "raw_kl_p99": float(group["raw_forward_kl_top128"].quantile(0.99)),
                "abs_gap_p50": float(group["abs_chosen_gap"].median()),
                "abs_gap_p90": float(group["abs_chosen_gap"].quantile(0.9)),
                "abs_gap_p99": float(group["abs_chosen_gap"].quantile(0.99)),
            }
        )
    return rows


def phase_summarize(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    shards = sorted((output_dir / "token_shards").glob("example_*.parquet"))
    if not shards:
        raise FileNotFoundError("No token shards found")
    frame = pd.concat((pd.read_parquet(path) for path in shards), ignore_index=True)
    frame.to_parquet(output_dir / "tokens.parquet", index=False, compression="zstd")
    rollouts = read_jsonl(output_dir / "rollouts.jsonl")
    metrics = [
        "chosen_logprob_gap",
        "abs_chosen_gap",
        "sampled_reverse_kl_k1",
        "sampled_reverse_kl_k3",
        "sampled_reverse_kl_k3_clipped20",
        "raw_forward_kl_top128",
        "effective_forward_kl_top128",
        "teacher_mass_top128",
        "student_mass_on_teacher_top128",
        "topk_overlap",
        "top1_agreement",
        "student_entropy",
        "teacher_entropy_lower_bound",
    ]
    summary: dict[str, Any] = {
        "dataset": rollouts[0]["dataset"],
        "num_rollouts": len(rollouts),
        "num_tokens": int(len(frame)),
        "accuracy": float(np.mean([row["correct"] for row in rollouts])),
        "response_length": {
            "mean": float(np.mean([row["response_tokens"] for row in rollouts])),
            "median": float(np.median([row["response_tokens"] for row in rollouts])),
            "max": int(max(row["response_tokens"] for row in rollouts)),
        },
        "token_weighted": {metric: _quantiles(frame[metric]) for metric in metrics},
        "tail_fractions": {
            "abs_gap_gt_1": float((frame["abs_chosen_gap"] > 1).mean()),
            "abs_gap_gt_2": float((frame["abs_chosen_gap"] > 2).mean()),
            "abs_gap_gt_5": float((frame["abs_chosen_gap"] > 5).mean()),
            "abs_gap_gt_10": float((frame["abs_chosen_gap"] > 10).mean()),
            "raw_kl_gt_0_1": float((frame["raw_forward_kl_top128"] > 0.1).mean()),
            "raw_kl_gt_0_5": float((frame["raw_forward_kl_top128"] > 0.5).mean()),
            "raw_kl_gt_1": float((frame["raw_forward_kl_top128"] > 1).mean()),
            "raw_kl_gt_2": float((frame["raw_forward_kl_top128"] > 2).mean()),
            "raw_kl_gt_5": float((frame["raw_forward_kl_top128"] > 5).mean()),
        },
    }
    per_rollout = frame.groupby("dataset_index", observed=True)[metrics].mean()
    summary["rollout_weighted"] = {
        metric: _quantiles(per_rollout[metric]) for metric in metrics
    }
    frame["absolute_bin"] = (frame["response_position"] // 256).astype(int)
    frame["relative_bin"] = np.minimum((frame["relative_position"] * 20).astype(int), 19)
    summary["position_absolute_256"] = _position_summary(frame, "absolute_bin")
    summary["position_relative_20"] = _position_summary(frame, "relative_bin")
    summary["by_correctness"] = {
        str(bool(correct)).lower(): {
            "tokens": int(len(group)),
            "rollouts": int(group["dataset_index"].nunique()),
            "raw_kl": _quantiles(group["raw_forward_kl_top128"]),
            "abs_gap": _quantiles(group["abs_chosen_gap"]),
        }
        for correct, group in frame.groupby("correct", observed=True)
    }
    if frame["difficulty"].notna().any():
        summary["by_difficulty"] = {
            str(difficulty): {
                "tokens": int(len(group)),
                "rollouts": int(group["dataset_index"].nunique()),
                "raw_kl": _quantiles(group["raw_forward_kl_top128"]),
                "abs_gap": _quantiles(group["abs_chosen_gap"]),
                "top1_agreement_mean": float(group["top1_agreement"].mean()),
                "topk_overlap_mean": float(group["topk_overlap"].mean()),
            }
            for difficulty, group in frame.groupby("difficulty", observed=True)
        }
    if frame["topic"].notna().any():
        summary["by_topic"] = {
            str(topic): {
                "tokens": int(len(group)),
                "rollouts": int(group["dataset_index"].nunique()),
                "raw_kl_mean": float(group["raw_forward_kl_top128"].mean()),
                "abs_gap_mean": float(group["abs_chosen_gap"].mean()),
                "top1_agreement_mean": float(group["top1_agreement"].mean()),
                "topk_overlap_mean": float(group["topk_overlap"].mean()),
            }
            for topic, group in frame.groupby("topic", observed=True)
        }
    write_json(output_dir / "summary.json", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["generate", "teacher", "student", "summarize"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="aime24")
    parser.add_argument("--data", type=Path, default=Path(DEFAULT_DATA))
    parser.add_argument("--student", default=DEFAULT_STUDENT)
    parser.add_argument(
        "--tokenizer",
        help="Optional tokenizer path; defaults to --student for backward compatibility",
    )
    parser.add_argument("--teacher", default=DEFAULT_TEACHER)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prompt-style", choices=["aime", "deepmath"], default="aime")
    parser.add_argument("--scorer", choices=["aime", "math"], default="aime")
    parser.add_argument("--score-timeout", type=float, default=20.0)
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Render the Qwen chat template with enable_thinking=False",
    )
    return parser.parse_args()


def main() -> None:
    global TOPK, SEED, MAX_TOKENS, MAX_MODEL_LEN
    args = parse_args()
    TOPK = args.topk
    SEED = args.seed
    MAX_TOKENS = args.max_tokens
    MAX_MODEL_LEN = args.max_model_len
    phase = {
        "generate": phase_generate,
        "teacher": phase_teacher,
        "student": phase_student,
        "summarize": phase_summarize,
    }[args.phase]
    phase(args)


if __name__ == "__main__":
    main()
