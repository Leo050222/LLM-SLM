"""Freeze matched greedy weak-model rollouts for the prefix-handoff run."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path("/storage/yukaichengLab/wangxin/test_eiloeo")
OUT = ROOT / "opd_eval_prefix_handoff_weak_to_8b_20261005"
BASE17 = ROOT / "opd_eval_baselines/full_teacher_student_greedy_20260920"
BASE4_REMAINING = ROOT / "opd_eval_baselines/qwen3_greedy_remaining_20261005"
BASE4_HARD = ROOT / "opd_eval_hard/qwen3_greedy_hard_three_20261005"
MAIN_DATASETS = {
    "math500": 500,
    "olympiad_bench": 674,
    "bbh": 1080,
    "cruxeval_i": 800,
    "cruxeval_o": 800,
    "humaneval_plus": 164,
    "mbpp_plus": 378,
    "livecodebench": 175,
}


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path):
    for line in path.read_text().splitlines():
        if line.strip():
            value = json.loads(line)
            yield value.get("data", value)


def read_completed(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file() and path.suffix == ".jsonl":
        yield from read_jsonl(path)
        return
    if path.is_file():
        value = json.loads(path.read_text())
        if "status" in value:
            yield value
        return
    for child in sorted(path.iterdir()):
        if child.name.startswith("jsonl_compaction") or child.name.endswith(".tmp"):
            continue
        if child.suffix == ".jsonl":
            yield from read_jsonl(child)
        elif child.suffix == ".json":
            value = json.loads(child.read_text())
            if "status" in value:
                yield value


def read_scores(path: Path):
    if not path.exists():
        return {}
    values = list(read_completed(path))
    result = {}
    for value in values:
        key = normalized_key(value)
        correct = value.get("correct", value.get("is_correct"))
        if isinstance(correct, bool):
            result[key] = correct
    return result


def normalized_key(row):
    problem_id = row["problem_id"]
    # The historical 1.7B math prompts carry a dataset prefix while the
    # 4B/8B 2026-10-05 files use the raw Math500 id.
    if row["dataset"] == "math500" and problem_id.startswith("math500/"):
        problem_id = problem_id[len("math500/") :]
    return row["dataset"], problem_id


def key(row):
    return normalized_key(row)


def assert_unique(rows, label):
    keys = [key(row) for row in rows]
    assert len(keys) == len(set(keys)), f"duplicate {label} rows"


def main():
    weak17 = list(read_completed(BASE17 / "math_student_r0/completed/records-00000.jsonl"))
    weak17 += list(read_completed(BASE17 / "code_student_r0/completed/records-00000.jsonl"))
    weak17 = [row for row in weak17 if row.get("dataset") in MAIN_DATASETS]
    weak4 = list(read_completed(BASE4_REMAINING / "qwen3_4b/completed"))
    weak4 += list(read_completed(BASE4_HARD / "qwen3_4b/completed"))
    weak4 = [row for row in weak4 if row.get("dataset") in MAIN_DATASETS]
    assert_unique(weak17, "1.7B")
    assert_unique(weak4, "4B")
    assert Counter(row["dataset"] for row in weak17) == MAIN_DATASETS
    assert Counter(row["dataset"] for row in weak4) == MAIN_DATASETS

    weak4_by_key = {key(row): row for row in weak4}
    score17 = read_scores(BASE17 / "math_student_r0/scores/records-00000.jsonl")
    score17.update(read_scores(BASE17 / "code_student_r0/scores/records-00000.jsonl"))
    score4 = read_scores(BASE4_REMAINING / "qwen3_4b/scores")
    score4.update(read_scores(BASE4_HARD / "qwen3_4b/scores"))

    weak17.sort(key=lambda row: (row["dataset"], row["problem_id"]))
    rows = []
    for canonical_index, row17 in enumerate(weak17):
        row4 = weak4_by_key.get(key(row17))
        assert row4 is not None, key(row17)
        assert row17["prompt_token_ids"] == row4["prompt_token_ids"], key(row17)
        assert row17["problem"] == row4["problem"], key(row17)
        row = {
            "canonical_index": canonical_index,
            "dataset": row17["dataset"],
            "problem_id": row17["problem_id"],
            "problem": row17["problem"],
            "solution": row17.get("solution"),
            "metadata": row17.get("metadata"),
            "subset": row17.get("subset"),
            "prompt_token_ids": list(row17["prompt_token_ids"]),
            "weak17_response_token_ids": list(row17.get("response_token_ids", [])),
            "weak4_response_token_ids": list(row4.get("response_token_ids", [])),
            "weak17_finish_reason": row17.get("finish_reason"),
            "weak4_finish_reason": row4.get("finish_reason"),
            "weak17_correct": score17.get(key(row17)),
            "weak4_correct": score4.get(key(row17)),
            "source_17_dataset_index": row17.get("dataset_index"),
            "source_4_dataset_index": row4.get("dataset_index"),
        }
        assert row["weak17_response_token_ids"] is not None
        assert row["weak4_response_token_ids"] is not None
        rows.append(row)

    assert len(rows) == sum(MAIN_DATASETS.values()) == 4571
    assert Counter(row["dataset"] for row in rows) == MAIN_DATASETS
    assert all(row["weak17_correct"] is not None for row in rows), "missing 1.7B baseline scores"
    assert all(row["weak4_correct"] is not None for row in rows), "missing 4B baseline scores"

    OUT.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows).encode()
    destination = OUT / "inputs.jsonl"
    if destination.exists():
        assert destination.read_bytes() == payload, "frozen inputs changed"
    else:
        destination.write_bytes(payload)
    manifest = {
        "rows": len(rows),
        "datasets": MAIN_DATASETS,
        "models": {"weak17": "Qwen3-1.7B", "weak4": "Qwen3-4B", "strong": "Qwen3-8B"},
        "prefix_percentages": [10, 20, 30, 40, 50],
        "prefix_rule": "floor(weak_response_token_count * percentage / 100)",
        "generation": {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "n": 1, "max_context": 40960, "max_response_tokens": 40264},
        "inputs_sha256": digest_bytes(payload),
    }
    (OUT / "INPUTS_COMPLETE.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
