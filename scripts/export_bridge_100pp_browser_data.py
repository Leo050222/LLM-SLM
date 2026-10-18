#!/usr/bin/env python3
"""Stream a compact web-data archive for strict 0/8 -> 8/8 bridges.

This script is designed to run on the experiment host. It reads the immutable
rollout and score artifacts and writes a gzipped tar archive to stdout. The
archive normalizes shared prefix data so the eight direct rollouts are stored
once per prefix rather than once per qualifying bridge.
"""

from __future__ import annotations

import io
import json
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any


TRAIN = Path("/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train")
DIRECT = TRAIN / "student_from_anchor_prefix_qwen3_1_7b_n8_seed20260827"
AFTER_0 = TRAIN / "student_after_bridge_qwen3_1_7b_h256_n8_seed20260826"
AFTER_X8 = TRAIN / "student_after_bridge_qwen3_1_7b_h256_n8_x8_seed20260826"
AFTER_ROOTS = [AFTER_0] + [AFTER_X8 / f"replicate_{index:02d}" for index in range(1, 8)]
AGGREGATE = AFTER_X8 / "aggregate" / "bridge_scores_x8.jsonl"


def jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_rollout(text: str, score: dict[str, Any], replicate: int) -> dict[str, Any]:
    return {
        "replicate": replicate,
        "correct": bool(score["correct"]),
        "predictedAnswer": score.get("predicted_answer"),
        "targetAnswer": score.get("target_answer"),
        "tokenCount": int(score.get("completion_token_count", 0)),
        "finishReason": score.get("finish_reason"),
        "text": text,
    }


def question_from_prompt(prompt: str) -> str:
    marker = "<|im_start|>user\n"
    if marker in prompt:
        prompt = prompt.split(marker, 1)[1]
    if "<|im_end|>" in prompt:
        prompt = prompt.split("<|im_end|>", 1)[0]
    instruction = "Solve the following math problem. Show your reasoning and put your final answer inside \\boxed{}.\n\n"
    if prompt.startswith(instruction):
        prompt = prompt[len(instruction):]
    return prompt.strip()


def trace_path(root: Path, dataset_index: int, name: str) -> Path:
    return root / "traces" / f"trace_{dataset_index:05d}" / name


def tar_add_json(archive: tarfile.TarFile, name: str, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(payload))


def main() -> None:
    qualifying: list[dict[str, Any]] = []
    qualifying_keys: set[tuple[int, int]] = set()
    by_prefix: dict[int, list[int]] = defaultdict(list)
    aggregate_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for row in jsonl(AGGREGATE):
        if int(row["correct_count"]) != 8:
            continue
        key = (int(row["dataset_index"]), int(row["bridge_index"]))
        aggregate_by_key[key] = row

    direct_scores: dict[tuple[int, int], dict[str, Any]] = {}
    direct_counts: dict[int, int] = defaultdict(int)
    candidate_prefixes = {dataset_index for dataset_index, _ in aggregate_by_key}
    for row in jsonl(DIRECT / "analysis_correctness_length_single_job" / "scores.jsonl"):
        dataset_index = int(row["dataset_index"])
        if dataset_index not in candidate_prefixes:
            continue
        rollout_index = int(row["rollout_index"])
        direct_scores[(dataset_index, rollout_index)] = row
        direct_counts[dataset_index] += int(bool(row["correct"]))

    for key, row in aggregate_by_key.items():
        dataset_index, bridge_index = key
        if direct_counts[dataset_index] == 0:
            qualifying_keys.add(key)
            by_prefix[dataset_index].append(bridge_index)
            qualifying.append(row)

    if len(qualifying_keys) != 731 or len(by_prefix) != 275:
        raise RuntimeError(
            f"strict selection mismatch: {len(qualifying_keys)} bridges from {len(by_prefix)} prefixes"
        )

    after_scores: list[dict[tuple[int, int], dict[str, Any]]] = []
    for root in AFTER_ROOTS:
        selected: dict[tuple[int, int], dict[str, Any]] = {}
        score_path = root / "analysis_correctness_length_single_job" / "scores.jsonl"
        for row in jsonl(score_path):
            key = (int(row["dataset_index"]), int(row["rollout_index"]))
            if key in qualifying_keys:
                selected[key] = row
        if set(selected) != qualifying_keys:
            raise RuntimeError(f"missing selected score rows in {score_path}")
        after_scores.append(selected)

    index_records: list[dict[str, Any]] = []
    prefix_payloads: dict[int, dict[str, Any]] = {}
    bridge_payloads: dict[tuple[int, int], dict[str, Any]] = {}

    for dataset_index in sorted(by_prefix):
        direct_trace = read_json(trace_path(DIRECT, dataset_index, "student_from_anchor_prefix_n8.json"))
        direct_rollouts = []
        for rollout in direct_trace["rollouts"]:
            rollout_index = int(rollout["rollout_index"])
            score = direct_scores[(dataset_index, rollout_index)]
            direct_rollouts.append(
                compact_rollout(rollout["student_completion_text"], score, rollout_index)
            )
        if len(direct_rollouts) != 8 or any(item["correct"] for item in direct_rollouts):
            raise RuntimeError(f"direct 0/8 invariant failed for {dataset_index}")

        problem = question_from_prompt(direct_trace["prompt_text"])
        first_score = direct_scores[(dataset_index, 0)]
        prefix_payloads[dataset_index] = {
            "datasetIndex": dataset_index,
            "problemId": direct_trace["problem_id"],
            "topic": first_score.get("topic"),
            "difficulty": first_score.get("difficulty"),
            "question": problem,
            "responsePrefix": direct_trace["response_prefix_before_anchor_text"],
            "studentAnchor": direct_trace.get("student_anchor_token_text"),
            "targetAnswer": first_score.get("target_answer"),
            "directCorrectCount": 0,
            "directRollouts": direct_rollouts,
        }

        traces = [
            read_json(trace_path(root, dataset_index, "student_after_bridge_n8.json"))
            for root in AFTER_ROOTS
        ]
        for bridge_index in sorted(by_prefix[dataset_index]):
            after_rollouts = []
            bridge_text = None
            bridge_tokens = None
            for replicate, trace in enumerate(traces):
                rollout = trace["rollouts"][bridge_index]
                score = after_scores[replicate][(dataset_index, bridge_index)]
                bridge_text = bridge_text or rollout["source_bridge_text"]
                bridge_tokens = bridge_tokens or int(rollout["source_bridge_token_count"])
                after_rollouts.append(
                    compact_rollout(rollout["student_continuation_text"], score, replicate)
                )
            if len(after_rollouts) != 8 or not all(item["correct"] for item in after_rollouts):
                raise RuntimeError(f"after 8/8 invariant failed for {dataset_index}/{bridge_index}")

            key = (dataset_index, bridge_index)
            bridge_payloads[key] = {
                "datasetIndex": dataset_index,
                "bridgeIndex": bridge_index,
                "bridgeTokenCount": bridge_tokens,
                "bridgeText": bridge_text,
                "afterCorrectCount": 8,
                "afterRollouts": after_rollouts,
            }
            index_records.append({
                "id": f"{dataset_index}-{bridge_index}",
                "datasetIndex": dataset_index,
                "bridgeIndex": bridge_index,
                "problemId": direct_trace["problem_id"],
                "topic": first_score.get("topic"),
                "difficulty": first_score.get("difficulty"),
                "questionPreview": problem[:220],
                "bridgePreview": (bridge_text or "")[:220],
                "qualifyingBridgesForPrefix": len(by_prefix[dataset_index]),
            })

    index_payload = {
        "schemaVersion": 1,
        "definition": "Direct Student from the same prefix is 0/8 correct and Student after this 256-token bridge is 8/8 correct.",
        "bridgeCount": len(index_records),
        "prefixCount": len(prefix_payloads),
        "replicatesPerCondition": 8,
        "records": index_records,
    }

    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as archive:
        tar_add_json(archive, "data/index.json", index_payload)
        for dataset_index, payload in prefix_payloads.items():
            tar_add_json(archive, f"data/prefixes/{dataset_index}.json", payload)
        for (dataset_index, bridge_index), payload in bridge_payloads.items():
            tar_add_json(archive, f"data/bridges/{dataset_index}-{bridge_index}.json", payload)


if __name__ == "__main__":
    main()
