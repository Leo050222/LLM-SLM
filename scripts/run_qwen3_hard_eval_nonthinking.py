#!/usr/bin/env python3
"""Run the fixed hard suite with Qwen3 thinking explicitly disabled."""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

from evalscope import TaskConfig, run_task


DATA_ROOT = Path(
    "/storage/yukaichengLab/wangxin/test_eiloeo/"
    "opd_eval_hard/modelscope_cache"
)

SYMBOLIC_EQUAL_TIMEOUT_SECONDS = 30.0


def _symbolic_equal_worker(a: str, b: str, connection) -> None:
    """Run EvalScope's unmodified symbolic equality check in isolation."""
    try:
        from evalscope.metrics.math import parser as math_parser

        connection.send(bool(math_parser.symbolic_equal(a, b)))
    except BaseException:
        connection.send(False)
    finally:
        connection.close()


def install_symbolic_equal_timeout() -> None:
    """Prevent pathological SymPy expressions from blocking an entire eval."""
    from evalscope.metrics.math import parser as math_parser

    context = mp.get_context("spawn")

    def symbolic_equal_with_timeout(a: str, b: str) -> bool:
        if str(a).strip() == str(b).strip():
            return True

        parent_connection, child_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=_symbolic_equal_worker,
            args=(a, b, child_connection),
            daemon=True,
        )
        process.start()
        child_connection.close()
        try:
            if parent_connection.poll(SYMBOLIC_EQUAL_TIMEOUT_SECONDS):
                return bool(parent_connection.recv())
            return False
        except (EOFError, OSError):
            return False
        finally:
            parent_connection.close()
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)

    math_parser.symbolic_equal = symbolic_equal_with_timeout


def dataset_args() -> dict:
    return {
        "aime24": {
            "local_path": str(DATA_ROOT / "evalscope/aime24"),
            "subset_list": ["default"],
            "few_shot_num": 0,
        },
        "aime25": {
            "local_path": str(DATA_ROOT / "evalscope/aime25"),
            "subset_list": ["default"],
            "few_shot_num": 0,
        },
        "hmmt25": {
            "local_path": str(DATA_ROOT / "evalscope/hmmt_feb_2025"),
            "subset_list": ["default"],
            "few_shot_num": 0,
        },
        "olympiad_bench": {
            "local_path": str(DATA_ROOT / "AI-ModelScope/OlympiadBench"),
            "subset_list": ["OE_TO_maths_en_COMP"],
        },
        "gpqa_diamond": {
            "local_path": str(DATA_ROOT / "AI-ModelScope/gpqa_diamond"),
            "few_shot_num": 0,
        },
        "live_code_bench": {
            "local_path": str(
                DATA_ROOT
                / "evalscope/livecodebench_code_generation_lite_parquet/v6"
            ),
            "subset_list": ["default"],
            "few_shot_num": 0,
            "extra_params": {"debug": False},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    install_symbolic_equal_timeout()

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
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False}
            },
        },
        judge_strategy="rule",
        work_dir=args.work_dir,
        use_cache=args.work_dir if args.resume else None,
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
