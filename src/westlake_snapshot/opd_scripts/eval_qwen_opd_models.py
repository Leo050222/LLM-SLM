#!/usr/bin/env python3
"""Greedy verl-compatible GSM8K/MATH evaluation with per-example traces."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.gsm8k import extract_solution as extract_gsm8k
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


def extracted_answer(data_source: str, response: str):
    if data_source == "openai/gsm8k":
        return extract_gsm8k(response, method="strict")
    boxed = last_boxed_only_string(response)
    if boxed is None:
        return None
    try:
        return remove_boxed(boxed)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tp", type=int, default=1)
    args = parser.parse_args()

    dataset = load_dataset("parquet", data_files=args.data, split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = [
        tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        for row in dataset
    ]

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        gpu_memory_utilization=0.80,
        max_model_len=4097,
        seed=20260806,
        trust_remote_code=False,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=2048, n=1, seed=20260806)
    outputs = llm.generate(prompts, sampling, use_tqdm=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    correct: dict[str, int] = {}
    length_finished: dict[str, int] = {}

    with output_path.open("w", encoding="utf-8") as handle:
        for row, request_output in zip(dataset, outputs, strict=True):
            generated = request_output.outputs[0]
            source = row["data_source"]
            ground_truth = row["reward_model"]["ground_truth"]
            score = float(default_compute_score(source, generated.text, ground_truth, row.get("extra_info")))
            counts[source] = counts.get(source, 0) + 1
            correct[source] = correct.get(source, 0) + int(score == 1.0)
            length_finished[source] = length_finished.get(source, 0) + int(generated.finish_reason == "length")
            record = {
                "model_name": args.model_name,
                "data_source": source,
                "index": row["extra_info"]["index"],
                "prompt": row["prompt"],
                "ground_truth": ground_truth,
                "response": generated.text,
                "extracted_answer": extracted_answer(source, generated.text),
                "score": score,
                "finish_reason": generated.finish_reason,
                "output_tokens": len(generated.token_ids),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "model_name": args.model_name,
        "model": args.model,
        "temperature": 0.0,
        "do_sample": False,
        "max_tokens": 2048,
        "results": {
            source: {
                "correct": correct[source],
                "total": counts[source],
                "accuracy": correct[source] / counts[source],
                "length_finished": length_finished[source],
            }
            for source in counts
        },
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("EVAL_OK", json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
