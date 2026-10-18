#!/usr/bin/env python3
"""Run the fixed hard-benchmark suite against a local OpenAI-compatible server."""

from __future__ import annotations

import argparse
from pathlib import Path

from evalscope import TaskConfig, run_task


DATA_ROOT = Path(
    "/storage/yukaichengLab/wangxin/test_eiloeo/"
    "opd_eval_hard/modelscope_cache"
)


def dataset_args() -> dict:
    final_answer_filter = {"remove_until": "</think>"}
    return {
        "aime24": {
            "local_path": str(DATA_ROOT / "evalscope/aime24"),
            "subset_list": ["default"],
            "few_shot_num": 0,
            "filters": final_answer_filter,
        },
        "aime25": {
            "local_path": str(DATA_ROOT / "evalscope/aime25"),
            "subset_list": ["default"],
            "few_shot_num": 0,
            "filters": final_answer_filter,
        },
        "hmmt25": {
            "local_path": str(DATA_ROOT / "evalscope/hmmt_feb_2025"),
            "subset_list": ["default"],
            "few_shot_num": 0,
            "filters": final_answer_filter,
        },
        "olympiad_bench": {
            "local_path": str(DATA_ROOT / "AI-ModelScope/OlympiadBench"),
            # Text-only English open-ended competition math is the largest
            # automatically scoreable subset for a text-only language model.
            "subset_list": ["OE_TO_maths_en_COMP"],
            "filters": final_answer_filter,
        },
        "gpqa_diamond": {
            "local_path": str(DATA_ROOT / "AI-ModelScope/gpqa_diamond"),
            "few_shot_num": 0,
            "filters": final_answer_filter,
        },
        "live_code_bench": {
            # The downloaded v6 incremental window contains 175 problems.
            # It is exposed as the local default config because only that
            # parquet shard is needed for this run.
            "local_path": str(
                DATA_ROOT
                / "evalscope/livecodebench_code_generation_lite_parquet/v6"
            ),
            "subset_list": ["default"],
            "few_shot_num": 0,
            "filters": final_answer_filter,
            "extra_params": {"debug": False},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config = TaskConfig(
        model=args.model_id,
        model_id=args.model_id,
        api_url=args.api_url,
        api_key="EMPTY",
        eval_type="openai_api",
        datasets=[
            "aime24",
            "aime25",
            "hmmt25",
            "olympiad_bench",
            "gpqa_diamond",
            "live_code_bench",
        ],
        dataset_args=dataset_args(),
        dataset_hub="local",
        repeats=1,
        limit=args.limit,
        eval_batch_size=16,
        generation_config={
            "max_tokens": 38912,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 1.5,
            "n": 1,
            "seed": 20260807,
            "stream": False,
            "timeout": 7200,
        },
        judge_strategy="rule",
        work_dir=args.work_dir,
        no_timestamp=True,
        enable_progress_tracker=True,
        ignore_errors=False,
        seed=20260807,
        collect_perf=True,
        use_sandbox=False,
    )
    run_task(task_cfg=config)


if __name__ == "__main__":
    main()
