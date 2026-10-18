#!/usr/bin/env python3
"""Score one greedy output directory with the frozen hard_three rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

N = 1814


def score_answer(row: dict) -> tuple[bool, str | None]:
    response = row["response"].rsplit("</think>", 1)[-1].strip()
    dataset = row["dataset"]
    if dataset == "bbh":
        from evalscope.benchmarks.bbh.bbh_adapter import BBHAdapter

        pred = (
            BBHAdapter._extract_mc_answer(response)
            if row["metadata"]["multiple_choice"]
            else BBHAdapter._extract_ff_answer(response)
        )
        canonical = (
            (lambda x: "".join(x.split()))
            if row["subset"] == "dyck_languages"
            else (lambda x: " ".join(x.split()).casefold())
        )
        return canonical(pred) == canonical(row["solution"]), pred
    if dataset in {"aime24", "aime25"}:
        from evalscope.benchmarks.aime.grader import grade_answer
        from evalscope.benchmarks.aime.math_normalize import normalize_answer
        from evalscope.metrics.math.parser import extract_answer

        pred = normalize_answer(extract_answer(response))
        return bool(grade_answer(pred, row["solution"])), pred
    if dataset == "olympiad_bench":
        from types import SimpleNamespace

        from evalscope.benchmarks.olympiad_bench.olympiad_bench_adapter import OlympiadBenchAdapter
        from evalscope.benchmarks.olympiad_bench.utils import MathJudger

        metadata = row["metadata"]
        pred = OlympiadBenchAdapter.extract_answer(
            None, response, SimpleNamespace(metadata=metadata)
        )
        precision = 1e-8
        if "Tuple" not in metadata["answer_type"] and metadata.get("error"):
            error = str(metadata["error"])
            precision = (
                [float(x) if x else 1e-8 for x in error.split(",")]
                if "," in error
                else float(error)
            )
        return bool(MathJudger().judge(pred, metadata["final_answer"][0], precision)), pred
    raise ValueError(dataset)


def one(arg: tuple[Path, int, Path]) -> dict:
    source_dir, index, score_dir = arg
    source = source_dir / "completed" / f"{index:05d}.json"
    row = json.loads(source.read_text())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    dest = score_dir / f"{index:05d}.json"
    if dest.exists():
        old = json.loads(dest.read_text())
        if old.get("source_sha256") == digest:
            return old
    status, correct, predicted = "ok", False, None

    def timeout(*_):
        raise TimeoutError()

    signal.signal(signal.SIGALRM, timeout)
    try:
        signal.alarm(30)
        correct, predicted = score_answer(row)
    except TimeoutError:
        status = "timeout"
    except Exception as exc:
        status = type(exc).__name__
    finally:
        signal.alarm(0)
    result = {
        "model": row["model"],
        "dataset": row["dataset"],
        "subset": row["subset"],
        "dataset_index": index,
        "problem_id": row["problem_id"],
        "correct": bool(correct),
        "predicted": predicted,
        "status": status,
        "source_sha256": digest,
        "tokens": len(row.get("response_token_ids", [])),
    }
    score_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False))
    tmp.replace(dest)
    return result


def run(args: argparse.Namespace) -> None:
    source_dir = Path(args.output_root)
    completed = list((source_dir / "completed").glob("*.json"))
    if len(completed) != N:
        raise RuntimeError(f"expected {N} completed rows, found {len(completed)}")
    score_dir = source_dir / "scores"
    tasks = [(source_dir, i, score_dir) for i in range(N)]
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if done % 100 == 0 or done == N:
                print(f"SCORED {done}/{N}", flush=True)
    groups: dict[str, list[dict]] = defaultdict(list)
    for result in results:
        groups[result["dataset"]].append(result)
    summary = {}
    for dataset, group in sorted(groups.items()):
        summary[dataset] = {
            "questions": len(group),
            "correct": sum(x["correct"] for x in group),
            "accuracy": sum(x["correct"] for x in group) / len(group),
            "score_status": dict(Counter(x["status"] for x in group)),
        }
    subtasks = {}
    bbh = [x for x in results if x["dataset"] == "bbh"]
    for subset in sorted({x["subset"] for x in bbh}):
        group = [x for x in bbh if x["subset"] == subset]
        subtasks[subset] = {
            "questions": len(group),
            "correct": sum(x["correct"] for x in group),
            "accuracy": sum(x["correct"] for x in group) / len(group),
        }
    (source_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (source_dir / "bbh_subtasks.json").write_text(json.dumps(subtasks, indent=2))
    (source_dir / "SCORING_COMPLETE").touch()
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, default=48)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
