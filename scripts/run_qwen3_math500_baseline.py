#!/usr/bin/env python3
"""Evaluate the original Qwen3-1.7B on the fixed local MATH-500 split."""

from __future__ import annotations

import argparse

from evalscope import TaskConfig, run_task


DATA_DIR = (
    "/storage/yukaichengLab/wangxin/test_eiloeo/"
    "opd_eval_math500/data/AI-ModelScope_MATH-500"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()

    config = TaskConfig(
        model=args.model_id,
        model_id=args.model_id,
        api_url=args.api_url,
        api_key="EMPTY",
        eval_type="openai_api",
        datasets=["math_500"],
        dataset_args={
            "math_500": {
                "local_path": DATA_DIR,
                "subset_list": [
                    "Level 1",
                    "Level 2",
                    "Level 3",
                    "Level 4",
                    "Level 5",
                ],
                "few_shot_num": 0,
                "filters": {"remove_until": "</think>"},
            }
        },
        dataset_hub="local",
        repeats=1,
        eval_batch_size=16,
        generation_config={
            "max_tokens": 40264,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "n": 1,
            "seed": 20260811,
            "stream": False,
            "timeout": 7200,
        },
        judge_strategy="rule",
        work_dir=args.work_dir,
        no_timestamp=True,
        enable_progress_tracker=True,
        ignore_errors=False,
        seed=20260811,
        collect_perf=True,
        use_sandbox=False,
    )
    run_task(task_cfg=config)


if __name__ == "__main__":
    main()
