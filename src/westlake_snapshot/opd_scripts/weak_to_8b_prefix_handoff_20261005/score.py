"""Score every prefix-handoff condition with the existing benchmark protocol."""
from __future__ import annotations

import collections
import concurrent.futures
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path("/storage/yukaichengLab/wangxin/test_eiloeo")
OUT = ROOT / "opd_eval_prefix_handoff_weak_to_8b_20261005"
SCRIPT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "eval_repos/evalscope"))
from code_scorer import extract, sandbox
import judge_client
from evalscope.benchmarks.bbh.bbh_adapter import BBHAdapter
from evalscope.benchmarks.olympiad_bench.olympiad_bench_adapter import OlympiadBenchAdapter
from evalscope.metrics.math.parser import extract_answer

ARMS = ("weak17_to_8b", "weak4_to_8b")
PERCENTAGES = (10, 20, 30, 40, 50)
DATASETS = ("math500", "olympiad_bench", "bbh", "cruxeval_i", "cruxeval_o", "humaneval_plus", "mbpp_plus", "livecodebench")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temp, path)


def load_code_data():
    data_dir = ROOT / "opd_eval_code/code_reasoning_grid_20260913/data"
    result = {}
    for ds, name, key in (
        ("cruxeval_o", "cruxeval.jsonl", "id"),
        ("cruxeval_i", "cruxeval.jsonl", "id"),
        ("humaneval_plus", "HumanEvalPlus.jsonl", "task_id"),
        ("mbpp_plus", "MbppPlus.jsonl", "task_id"),
        ("livecodebench", "livecodebench_v6_incremental.jsonl", "question_id"),
    ):
        items = [json.loads(line) for line in (data_dir / name).read_text().splitlines()]
        result[ds] = {(item["platform"] + "/" + item[key] if ds == "livecodebench" else item[key]): item for item in items}
    return result


def candidate(row):
    response = row["response"].rsplit("</think>", 1)[-1].strip()
    ds = row["dataset"]
    if ds == "math500":
        return extract_answer(response)
    if ds == "bbh":
        metadata = row.get("metadata") or {}
        extractor = BBHAdapter._extract_mc_answer if metadata.get("multiple_choice") else BBHAdapter._extract_ff_answer
        return extractor(response)
    if ds == "olympiad_bench":
        return OlympiadBenchAdapter.extract_answer(None, response, SimpleNamespace(metadata=row.get("metadata") or {}))
    raise ValueError(ds)


def score_condition(arm, percentage, rows, code_data, client, judge_prompt):
    source_dir = OUT / arm / f"cap{percentage}" / "completed"
    files = sorted(source_dir.glob("*.json"))
    assert len(files) == len(rows) == 4571, (arm, percentage, len(files))
    score_dir = OUT / arm / f"cap{percentage}" / "scores"
    signature = hashlib.sha256(judge_prompt.encode()).hexdigest()
    input_by_index = {row["canonical_index"]: row for row in rows}

    def action(path):
        destination = score_dir / path.name
        source_hash = sha(path)
        if destination.exists():
            saved = json.loads(destination.read_text())
            assert saved["source_sha256"] == source_hash
            return saved
        output = json.loads(path.read_text())
        source_input = input_by_index[output["canonical_index"]]
        output.setdefault("solution", source_input.get("solution"))
        output.setdefault("metadata", source_input.get("metadata"))
        ds = output["dataset"]
        if ds in code_data:
            try:
                result = sandbox(dict(dataset=ds, problem=code_data[ds][output["problem_id"]], solution=extract(ds, output["response"])))
            except Exception as exc:
                result = {"correct": False, "status": "harness_error:" + type(exc).__name__}
        elif ds == "math500":
            try:
                predicted = candidate(output)
                reference = output["solution"]
                from evalscope.metrics.math.parser import math_equal, strip_answer_string

                result = {"correct": bool(predicted) and bool(math_equal(predicted, strip_answer_string(reference))), "status": "ok", "predicted": predicted, "reference": reference}
            except Exception as exc:
                result = {"correct": False, "status": "parser_error:" + type(exc).__name__}
        else:
            predicted = candidate(output)
            reference = (output.get("metadata") or {}).get("final_answer", [None])[0] if ds == "olympiad_bench" else output.get("solution")
            if predicted is None or not str(predicted).strip():
                result = {"correct": False, "is_correct": False, "status": "extraction_missing", "reason": "empty extracted answer"}
            else:
                user = json.dumps({"candidate_answer": predicted, "reference_answer": reference}, ensure_ascii=False)
                result = client.call(user)
                result.update(candidate_answer=predicted, reference_answer=reference, input_sha256=hashlib.sha256(user.encode()).hexdigest())
        result.update(
            dataset=ds,
            problem_id=output["problem_id"],
            canonical_index=output["canonical_index"],
            prefix_percentage=percentage,
            arm=arm,
            weak_correct=output["weak_correct"],
            finish_reason=output["finish_reason"],
            source_sha256=source_hash,
            judge_signature=signature,
            scored_at=time.time(),
        )
        atomic(destination, result)
        return result

    pending = files
    for attempt in range(3):
        errors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as pool:
            futures = {pool.submit(action, path): path for path in pending}
            for number, future in enumerate(concurrent.futures.as_completed(futures), 1):
                try:
                    result = future.result()
                    if result.get("status") == "api_error":
                        errors.append(futures[future])
                except Exception:
                    errors.append(futures[future])
                if number % 500 == 0 or number == len(pending):
                    print(f"SCORE_PROGRESS arm={arm} cap={percentage} attempt={attempt + 1} done={number}/{len(pending)} api_errors={len(errors)}", flush=True)
        if not errors:
            break
        pending = errors
        time.sleep(30)
    else:
        raise RuntimeError(f"unresolved scoring errors for {arm} cap{percentage}: {len(pending)}")

    results = [json.loads(path.read_text()) for path in sorted(score_dir.glob("*.json"))]
    assert len(results) == 4571
    groups = collections.defaultdict(list)
    for result in results:
        groups[result["dataset"]].append(result)
    summary = []
    for ds in DATASETS:
        group = groups[ds]
        correct = sum(bool(item["correct"]) for item in group)
        weak_correct = sum(bool(item["weak_correct"]) for item in group)
        summary.append({
            "dataset": ds,
            "n": len(group),
            "correct": correct,
            "accuracy": correct / len(group),
            "weak_correct": weak_correct,
            "weak_accuracy": weak_correct / len(group),
            "delta_pp_vs_weak": 100 * (correct - weak_correct) / len(group),
            "rescued": sum(bool(item["correct"]) and not bool(item["weak_correct"]) for item in group),
            "harmed": sum(not bool(item["correct"]) and bool(item["weak_correct"]) for item in group),
            "score_status": dict(collections.Counter(str(item.get("status", "unknown")) for item in group)),
        })
    total_correct = sum(item["correct"] for item in summary)
    total_weak = sum(item["weak_correct"] for item in summary)
    result = {
        "arm": arm,
        "prefix_percentage": percentage,
        "rows": len(results),
        "correct": total_correct,
        "accuracy": total_correct / len(results),
        "weak_correct": total_weak,
        "weak_accuracy": total_weak / len(results),
        "delta_pp_vs_weak": 100 * (total_correct - total_weak) / len(results),
        "rescued": sum(item["rescued"] for item in summary),
        "harmed": sum(item["harmed"] for item in summary),
        "groups": summary,
    }
    atomic(OUT / arm / f"cap{percentage}" / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    rows = [json.loads(line) for line in (OUT / "inputs.jsonl").read_text().splitlines() if line.strip()]
    assert len(rows) == 4571
    code_data = load_code_data()
    judge_dir = OUT / "judge"
    judge_prompt = (judge_dir / "system_prompt.txt").read_text()
    relay = json.loads((judge_dir / "relay.json").read_text())
    client = judge_client.Client(int(relay["port"]), 64, judge_prompt)
    all_results = []
    for arm in ARMS:
        for percentage in PERCENTAGES:
            all_results.append(score_condition(arm, percentage, rows, code_data, client, judge_prompt))
            atomic(OUT / "summary.json", {"conditions": all_results, "updated_at": time.time()})
    atomic(OUT / "ALL_SCORING_COMPLETE.json", {"conditions": len(all_results), "rows": 45710, "completed_at": time.time()})
    print("ALL_SCORING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
