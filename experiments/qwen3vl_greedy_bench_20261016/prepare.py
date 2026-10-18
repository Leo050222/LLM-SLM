#!/usr/bin/env python3
"""Prepare the five public, labeled visual benchmark splits entirely offline.

Example:
  python prepare.py --benchmark-root /path/to/benchmarks/vision --output-dir RUN/prepared

The original image bytes are materialized once, with content hashes and dimensions.
Ground truth is retained in the task record but never enters prompt_messages.
Dependencies: pyarrow, Pillow, and the downloaded root/load_benchmark.py.
"""

import argparse
from collections import Counter
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import time


SCHEMA_VERSION = "qwen3vl-public-eval-v1"
EXPECTED_COUNTS = {"MathVista": 1000, "MathVision": 3040, "CV-Bench": 2638,
                   "BLINK": 1901, "RealWorldQA": 765}
SPLITS = {"MathVista": "testmini", "MathVision": "test", "CV-Bench": None,
          "BLINK": "val", "RealWorldQA": "test"}


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(value).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"sha256": digest(bytes(value)), "bytes": len(value)}
    return value


def load_offline_helper(root):
    spec = importlib.util.spec_from_file_location("offline_visual_benchmarks", root / "load_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_rows(root, helper, benchmark, split):
    """Retain exact source file and zero-based source row, including BLINK task."""
    import pyarrow.parquet as pq
    dataset, files = helper.selected_files(root, benchmark, split=split)
    for file_info in sorted(files, key=lambda x: x["path"]):
        path = helper.data_path(root, dataset["name"], file_info["path"])
        if path.suffix != ".parquet":
            raise ValueError("This evaluation expects Parquet input")
        row_index = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                yield dataset, file_info, row_index, row
                row_index += 1


def save_image(value, root, output_dir):
    from PIL import Image
    if isinstance(value, dict):
        blob = value.get("bytes")
        if blob is None and value.get("path"):
            source = (root / value["path"]).resolve()
            if not source.is_relative_to(root):
                raise ValueError("Image path escaped the benchmark directory")
            blob = source.read_bytes()
    else:
        blob = value if isinstance(value, (bytes, bytearray, memoryview)) else None
    if not blob:
        raise ValueError("Image is not present as offline bytes")
    blob = bytes(blob)
    checksum = digest(blob)
    with Image.open(io.BytesIO(blob)) as image:
        width, height = image.size
        image_format = image.format.lower()
        image.verify()
    suffix = {"jpeg": "jpg", "tiff": "tif"}.get(image_format, image_format)
    path = output_dir / "images" / checksum[:2] / f"{checksum}.{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if digest(path.read_bytes()) != checksum:
            raise ValueError("An existing materialized image has changed")
    else:
        temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        temporary.write_bytes(blob)
        os.replace(temporary, path)
    return {"path": str(path), "sha256": checksum, "width": width, "height": height,
            "format": image_format, "bytes": len(blob)}


def choices_from_question(question):
    matches = list(re.finditer(r"(?m)^\s*(?:\(([A-Z])\)|([A-Z])[.)])\s+(.+?)\s*$", question))
    return [{"label": m.group(1) or m.group(2), "text": m.group(3)} for m in matches]


def canonical_choices(row):
    original = row.get("choices", row.get("options")) or []
    if isinstance(original, dict):
        return [{"label": str(k).strip("(). "), "text": str(v)} for k, v in original.items()]
    if original:
        result = []
        for index, value in enumerate(original):
            label = chr(ord("A") + index)
            text = str(value)
            # Strip only an explicit matching enumeration, never mathematical brackets.
            prefix = re.match(r"^\s*(?:\(" + label + r"\)|" + label + r"[.)])\s+", text)
            result.append({"label": label, "text": text[prefix.end():] if prefix else text})
        return result
    return choices_from_question(str(row.get("question") or ""))


def canonical_answer(raw, choices):
    if raw is None or not str(raw).strip():
        raise ValueError("A selected public evaluation row has no reference answer")
    text = str(raw).strip()
    labels = {option["label"]: option["text"] for option in choices}
    label = text.strip("(). ")
    if label not in labels:
        matches = [key for key, option in labels.items() if option.strip() == text]
        label = matches[0] if len(matches) == 1 else None
    return {"raw": json_value(raw), "label": label, "text": labels[label] if label else text}


def prompt_text(benchmark, row, choices):
    if benchmark == "MathVista":
        text = str(row.get("query") or row["question"]).strip()
    elif benchmark in ("CV-Bench", "BLINK"):
        text = str(row.get("prompt") or row["question"]).strip()
    else:
        text = str(row["question"]).strip()
    # MathVision uses a plain image marker; actual image content is inserted separately.
    text = re.sub(r"<image\d*>", "", text).strip()
    if choices:
        # Official CV-Bench/BLINK/MathVista prompts already include choices.
        present = all(re.search(r"(?:\(" + re.escape(option["label"]) + r"\)|(?m:^\s*" +
                                re.escape(option["label"]) + r"[.)]))\s*", text)
                      for option in choices)
        if not present:
            text += "\n\nChoices:\n" + "\n".join(f"({option['label']}) {option['text']}" for option in choices)
        text += "\n\nAnswer with only the letter of the correct option."
    else:
        text += "\n\nSolve the problem and end your response with 'Final Answer: <answer>'."
    return text


def build_task(root, output_dir, benchmark, dataset, file_info, source_index, row, index):
    question = str(row.get("question") or "").strip()
    if not question:
        raise ValueError("Empty question")
    choices = canonical_choices(row)
    answer = canonical_answer(row.get("answer"), choices)
    if benchmark == "BLINK":
        image_keys = [key for key in ("image_1", "image_2", "image_3", "image_4") if row.get(key)]
        subtask = Path(file_info["path"]).parts[0]
        split = "val"
    else:
        image_keys = ["decoded_image"] if row.get("decoded_image") else ["image"]
        subtask = str(row.get("type") or "") if benchmark == "CV-Bench" else ""
        split = "test_" + subtask.lower() if benchmark == "CV-Bench" else SPLITS[benchmark]
    images = [save_image(row[key], root / benchmark, output_dir) for key in image_keys]
    if not images:
        raise ValueError("Visual evaluation row has no images")
    source_id = str(next((row[key] for key in ("pid", "id", "idx") if row.get(key) is not None), source_index))
    source = {"repo_id": dataset["repo_id"], "revision": dataset["revision"],
              "file": file_info["path"], "file_sha256": file_info.get("sha256"),
              "row_index": source_index, "row_id": source_id}
    text = prompt_text(benchmark, row, choices)
    identity = {"schema_version": SCHEMA_VERSION, "benchmark": benchmark, "split": split,
                "subtask": subtask, "source": source, "question": question,
                "choices": choices, "answer": answer,
                "images": [{k: image[k] for k in ("sha256", "width", "height")} for image in images],
                "prompt_text": text}
    identity_sha256 = digest(identity)
    safe_source_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_id)
    task_id = "/".join(part for part in (benchmark, split, subtask, safe_source_id + "-" + identity_sha256[:12]) if part)
    nonimage = {key: json_value(value) for key, value in row.items() if key not in image_keys}
    return {"schema_version": SCHEMA_VERSION, "dataset_index": index, "task_id": task_id,
            "benchmark": benchmark, "split": split, "subtask": subtask,
            "source_row_id": source_id, "source": source, "question": question,
            "choices": choices, "answer": answer, "images": images,
            "prompt_messages": [{"role": "user", "content":
                                 [{"type": "image", "image": image["path"]} for image in images] +
                                 [{"type": "text", "text": text}]}],
            "metadata": {"source_nonimage": nonimage},
            "identity_sha256": identity_sha256}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmarks", nargs="+", choices=list(EXPECTED_COUNTS), default=list(EXPECTED_COUNTS))
    args = parser.parse_args()
    root = args.benchmark_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    helper = load_offline_helper(root)
    counts = Counter()
    subtasks = Counter()
    seen_ids = set()
    destination = output_dir / "tasks.jsonl"
    temporary = output_dir / f"tasks.jsonl.tmp.{os.getpid()}"
    # An interrupted rebuild must never leave an old success marker beside partial data.
    (output_dir / "PREPARED.json").unlink(missing_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        for benchmark in args.benchmarks:
            for dataset, file_info, source_index, row in source_rows(root, helper, benchmark, SPLITS[benchmark]):
                task = build_task(root, output_dir, benchmark, dataset, file_info, source_index, row, len(seen_ids))
                if task["task_id"] in seen_ids:
                    raise ValueError("Duplicate task ID")
                seen_ids.add(task["task_id"])
                handle.write(json.dumps(task, ensure_ascii=False, allow_nan=False) + "\n")
                counts[benchmark] += 1
                subtasks[f"{benchmark}/{task['split']}/{task['subtask']}"] += 1
            if counts[benchmark] != EXPECTED_COUNTS[benchmark]:
                raise ValueError(f"{benchmark} expected {EXPECTED_COUNTS[benchmark]} rows, found {counts[benchmark]}")
            print(json.dumps({"benchmark": benchmark, "prepared": counts[benchmark]}, ensure_ascii=False), flush=True)
    os.replace(temporary, destination)
    manifest = {"schema_version": SCHEMA_VERSION, "tasks": len(seen_ids), "counts": dict(counts),
                "subtasks": dict(subtasks), "tasks_sha256": digest(destination.read_bytes()),
                "benchmark_plan_sha256": digest((root / "plan.json").read_bytes()),
                "prepare_code_sha256": digest(Path(__file__).read_bytes()),
                "benchmark_root": str(root), "output_dir": str(output_dir),
                "image_policy": "original bytes; all images verified; no resize or image-count truncation",
                "ground_truth_policy": "reference retained in canonical rows; absent from prompt_messages",
                "completed_at": time.time()}
    atomic_json(output_dir / "PREPARED.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
