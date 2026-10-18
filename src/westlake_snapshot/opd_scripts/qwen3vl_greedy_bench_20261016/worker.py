#!/usr/bin/env python3
"""One offline vLLM replica per visible GPU for the prepared visual benchmarks.

Formal example (no artificial output cap):
  CUDA_VISIBLE_DEVICES=0 python worker.py --model-path /models/Qwen3-VL-4B-Instruct \
    --tasks /run/prepared/tasks.jsonl --output /run/results/Qwen3-VL-4B-Instruct \
    --shard-index 0 --num-shards 8 --batch-size 1

Startup smoke only, in a separate output directory:
  ... --output /run/smoke/4B --smoke-limit 5 --smoke-max-tokens 64

--preflight-only performs real CPU image processing and records prompt lengths;
it never constructs a GPU engine or writes completed inference records. No extra
qwen-vl-utils dependency is needed: the local official AutoProcessor is used.
"""

import argparse
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
import uuid


SCHEMA_VERSION = "qwen3vl-greedy-worker-v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    data = value if isinstance(value, bytes) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False,
                      default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def exclusive_lock(path, nonblocking=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
        fcntl.flock(handle, flags)
        yield


def emit(event, **values):
    print(json.dumps({"time": now(), "event": event, **values},
                     ensure_ascii=False, default=str), flush=True)


def read_tasks(path):
    tasks = []
    seen_ids, seen_indices = set(), set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            index = row.get("dataset_index", row.get("task_index"))
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise ValueError(f"Invalid task index on line {line_number}")
            row["dataset_index"] = index
            task_id = row["task_id"]
            if task_id in seen_ids or index in seen_indices:
                raise ValueError(f"Duplicate task ID/index on line {line_number}")
            if not row.get("identity_sha256") or not row.get("images"):
                raise ValueError(f"Missing task identity/images on line {line_number}")
            if not row.get("prompt_messages"):
                raise ValueError(f"Missing inference messages on line {line_number}")
            seen_ids.add(task_id)
            seen_indices.add(index)
            tasks.append(row)
    if not tasks:
        raise ValueError("The task file is empty")
    if [row["dataset_index"] for row in tasks] != sorted(seen_indices):
        raise ValueError("Tasks must retain increasing canonical dataset indices")
    return tasks


def select_tasks(tasks, shard_index, num_shards, smoke_limit=None):
    selected = [row for row in tasks if row["dataset_index"] % num_shards == shard_index]
    if smoke_limit is None:
        return selected
    # Exercise different benchmark schemas, including multi-image BLINK, early.
    groups = {}
    for row in selected:
        groups.setdefault(row["benchmark"], deque()).append(row)
    smoke = []
    while groups and len(smoke) < smoke_limit:
        for name in list(groups):
            smoke.append(groups[name].popleft())
            if not groups[name]:
                del groups[name]
            if len(smoke) == smoke_limit:
                break
    return sorted(smoke, key=lambda row: row["dataset_index"])


def local_model_identity(model_path):
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text())
    if config.get("model_type") != "qwen3_vl":
        raise ValueError("This worker only accepts dense Qwen3-VL models")
    if "thinking" in model_path.name.lower():
        raise ValueError("This run is restricted to Instruct checkpoints")
    maximum = config.get("text_config", {}).get("max_position_embeddings")
    if not isinstance(maximum, int) or maximum < 1:
        raise ValueError("No valid text_config.max_position_embeddings in local config")
    names = ("config.json", "generation_config.json", "preprocessor_config.json",
             "processor_config.json", "chat_template.json", "chat_template.jinja",
             "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "vocab.json", "merges.txt", "model.safetensors.index.json")
    files = {name: digest((model_path / name).read_bytes())
             for name in names if (model_path / name).is_file()}
    weights = [{"name": path.name, "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns}
               for path in sorted(model_path.glob("*.safetensors"))]
    if not weights:
        raise ValueError("No local safetensors weights are present")
    weight_index = model_path / "model.safetensors.index.json"
    if weight_index.exists():
        expected = set(json.loads(weight_index.read_text())["weight_map"].values())
        missing = expected - {item["name"] for item in weights}
        if missing:
            raise ValueError(f"Incomplete checkpoint, missing weight shards: {sorted(missing)}")
    return {"path": str(model_path), "configuration_files_sha256": files,
            "weight_file_inventory": weights, "max_model_len": maximum}, config


def validate_messages(row):
    """Only the prepared user prompt and ordered images reach the model."""
    messages = row["prompt_messages"]
    if len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError("Expected one user message and no supplied assistant/GT turns")
    image_paths = []
    for item in messages[0]["content"]:
        if item.get("type") == "image":
            if set(item) != {"type", "image"}:
                raise ValueError("Per-image processor overrides are not permitted")
            image_paths.append(str(Path(item["image"]).expanduser().resolve()))
        elif item.get("type") == "text":
            if set(item) != {"type", "text"} or not isinstance(item["text"], str):
                raise ValueError("Invalid text content")
        else:
            raise ValueError("Unsupported content in the image-only benchmark prompt")
    expected = [str(Path(item["path"]).expanduser().resolve()) for item in row["images"]]
    if image_paths != expected:
        raise ValueError("Message images differ from the canonical task image order")


def prepare_prompt(row, processor, maximum):
    from PIL import Image, ImageOps

    validate_messages(row)
    images = []
    for item in row["images"]:
        path = Path(item["path"]).expanduser().resolve()
        data = path.read_bytes()
        if digest(data) != item["sha256"]:
            raise ValueError(f"Image checksum mismatch: {path}")
        with Image.open(io.BytesIO(data)) as source:
            if list(source.size) != [item["width"], item["height"]]:
                raise ValueError(f"Image dimensions changed: {path}")
            # Same RGB pixels go through both HF and vLLM; no manual resizing.
            images.append(ImageOps.exif_transpose(source).convert("RGB"))
    prompt = processor.apply_chat_template(row["prompt_messages"], tokenize=False,
                                           add_generation_prompt=True)
    encoded = processor(text=[prompt], images=images, return_tensors="pt",
                        padding=False, truncation=False, add_special_tokens=False)
    input_ids = encoded["input_ids"][0].tolist()
    grid = encoded["image_grid_thw"].tolist()
    if len(grid) != len(images):
        raise ValueError("Processor lost or added images")
    image_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    image_tokens = sum(token == image_id for token in input_ids)
    merge_size = int(getattr(processor.image_processor, "merge_size", 2))
    grid_tokens = sum(int(t) * int(h) * int(w) // (merge_size ** 2) for t, h, w in grid)
    if image_tokens != grid_tokens:
        raise ValueError(f"Image token/grid mismatch: {image_tokens} != {grid_tokens}")
    prompt_tokens = len(input_ids)
    maximum_output = maximum - prompt_tokens
    if not input_ids or maximum_output <= 0:
        raise ValueError(f"Prompt length {prompt_tokens} leaves no output space in {maximum}")
    stats = {"prompt_tokens": prompt_tokens, "image_tokens": image_tokens,
             "non_image_tokens": prompt_tokens - image_tokens,
             "image_count": len(images), "image_grid_thw": grid,
             "processor_prompt_ids_sha256": digest(input_ids),
             "prompt_text_sha256": digest(prompt.encode()),
             "context_max_tokens": maximum, "context_remaining_tokens": maximum_output}
    # Do not keep the large CPU pixel_values allocation while vLLM preprocesses.
    del encoded
    return {"prompt": prompt, "multi_modal_data": {"image": images}}, input_ids, stats


def check_previous(path, row, identity):
    if not path.exists():
        return False
    result = json.loads(path.read_text())
    expected = {"task_id": row["task_id"], "task_identity_sha256": row["identity_sha256"],
                "run_identity_sha256": identity, "status": "completed"}
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Existing result has another task/model/settings identity: {path}")
    return True


def base_record(row, args, identity):
    return {"schema_version": SCHEMA_VERSION, "task_id": row["task_id"],
            "dataset_index": row["dataset_index"], "benchmark": row["benchmark"],
            "split": row.get("split"), "subtask": row.get("subtask"),
            "task_identity_sha256": row["identity_sha256"],
            "model": args.model_path.name, "model_path": str(args.model_path),
            "run_identity_sha256": identity, "shard_index": args.shard_index,
            "num_shards": args.num_shards, "mode": "smoke" if args.smoke_limit else "formal",
            "host": socket.gethostname(), "pid": os.getpid()}


def save_error(row, args, identity, error, **extra):
    record = {**base_record(row, args, identity), **extra, "status": "error", "time": now(),
              "error_type": type(error).__name__, "error": str(error),
              "traceback": traceback.format_exc()}
    destination = args.output / "errors" / f"{row['dataset_index']:05d}.{uuid.uuid4().hex}.json"
    atomic_json(destination, record)
    emit("task_error", task_id=row["task_id"], error=str(error), path=str(destination))


def packages():
    values = {}
    for name in ("torch", "transformers", "vllm", "Pillow", "numpy", "tokenizers"):
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def run(args):
    # Local files only: a missing model dependency must fail, never download.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tasks = read_tasks(args.tasks)
    selected = select_tasks(tasks, args.shard_index, args.num_shards, args.smoke_limit)
    model_identity, _ = local_model_identity(args.model_path)
    maximum = model_identity["max_model_len"]
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(args.model_path), local_files_only=True,
                                              trust_remote_code=False)
    image_settings = processor.image_processor.to_dict()
    max_images = max(len(row["images"]) for row in tasks)
    engine_kwargs = {"model": str(args.model_path), "tensor_parallel_size": 1,
                     "dtype": "bfloat16", "max_model_len": maximum,
                     "max_num_seqs": args.batch_size,
                     "max_num_batched_tokens": args.max_num_batched_tokens,
                     "gpu_memory_utilization": args.gpu_memory_utilization,
                     "limit_mm_per_prompt": {"image": max_images, "video": 0},
                     "enable_chunked_prefill": True, "enable_prefix_caching": False,
                     "trust_remote_code": False, "seed": 0,
                     "generation_config": "vllm", "disable_log_stats": False,
                     "enforce_eager": args.enforce_eager}
    settings = {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
                "repetition_penalty": 1.0, "presence_penalty": 0.0,
                "frequency_penalty": 0.0, "n": 1, "seed": 0,
                "ignore_eos": False, "detokenize": True,
                "skip_special_tokens": True,
                "max_tokens_policy": "config.text_config.max_position_embeddings - expanded_prompt_tokens",
                "smoke_max_tokens": args.smoke_max_tokens}
    run_info = {"schema_version": SCHEMA_VERSION, "model": model_identity,
                "tasks_sha256": digest(args.tasks.read_bytes()), "tasks_count": len(tasks),
                "canonical_task_ids_sha256": digest([row["task_id"] for row in tasks]),
                "processor_class": type(processor).__name__,
                "processor_image_settings": image_settings,
                "chat_template_sha256": digest(processor.chat_template),
                "engine_kwargs": engine_kwargs, "sampling_settings": settings,
                "packages": packages(), "worker_sha256": digest(Path(__file__).read_bytes()),
                "mode": "smoke" if args.smoke_limit else "formal"}
    identity = digest(run_info)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / ("PREFLIGHT_RUN.json" if args.preflight_only else "RUN.json")
    with exclusive_lock(args.output / "locks" / "run.lock", nonblocking=False):
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous.get("run_identity_sha256") != identity:
                raise ValueError("Output directory belongs to another run; use a fresh directory")
        else:
            atomic_json(manifest_path, {"run_identity_sha256": identity, **run_info})
        other_mode = args.output / ("FORMAL_ROOT" if args.smoke_limit else "SMOKE_ROOT")
        if other_mode.exists():
            raise ValueError("Smoke and formal output directories must be separate")
        (args.output / ("SMOKE_ROOT" if args.smoke_limit else "FORMAL_ROOT")).touch()
    status_path = args.output / ("preflight" if args.preflight_only else "status") / f"shard_{args.shard_index:03d}.json"
    status = {"run_identity_sha256": identity, "shard_index": args.shard_index,
              "num_shards": args.num_shards, "selected_tasks": len(selected),
              "selected_task_ids_sha256": digest([row["task_id"] for row in selected]),
              "started_at": now(), "state": "preflighting" if args.preflight_only else "initializing",
              "completed": 0, "resumed": 0, "errors": 0,
              "host": socket.gethostname(), "pid": os.getpid()}
    atomic_json(status_path, status)
    emit("worker_start", **status)
    if args.preflight_only:
        prompt_counts = []
        for row in selected:
            try:
                _, _, stats = prepare_prompt(row, processor, maximum)
                prompt_counts.append(stats["prompt_tokens"])
                atomic_json(args.output / "preflight" / f"{row['dataset_index']:05d}.json",
                            {**base_record(row, args, identity), "status": "preflight_only",
                             "prompt_stats": stats})
                emit("preflight_ok", task_id=row["task_id"], **stats)
            except Exception as error:
                status["errors"] += 1
                save_error(row, args, identity, error, preflight_only=True)
        status.update(state="preflight_complete" if not status["errors"] else "preflight_failed",
                      finished_at=now(), checked=len(prompt_counts),
                      min_prompt_tokens=min(prompt_counts, default=None),
                      max_prompt_tokens=max(prompt_counts, default=None))
        atomic_json(status_path, status)
        return 2 if status["errors"] else 0

    todo = []
    for row in selected:
        destination = args.output / "completed" / f"{row['dataset_index']:05d}.json"
        if check_previous(destination, row, identity):
            status["resumed"] += 1
        else:
            todo.append(row)
    if not todo:
        status.update(state="generation_complete", finished_at=now())
        atomic_json(status_path, status)
        emit("worker_finished", **status)
        return 0
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU to each independent worker")
    gpu = {"name": torch.cuda.get_device_name(0),
           "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "torch_cuda_version": torch.version.cuda}
    engine = LLM(**engine_kwargs)
    # Engine initialization must genuinely accept the full context length.
    actual_maximum = engine.llm_engine.model_config.max_model_len
    if actual_maximum != maximum:
        raise RuntimeError(f"Engine changed context length: {actual_maximum} != {maximum}")
    status.update(state="generating", gpu=gpu, vllm_version=vllm.__version__,
                  engine_initialized_at=now())
    atomic_json(status_path, status)
    for offset in range(0, len(todo), args.batch_size):
        batch, requests, expected_ids, prompt_stats, parameters = [], [], [], [], []
        for row in todo[offset:offset + args.batch_size]:
            try:
                request, ids, stats = prepare_prompt(row, processor, maximum)
                maximum_output = stats["context_remaining_tokens"]
                if args.smoke_max_tokens is not None:
                    maximum_output = min(maximum_output, args.smoke_max_tokens)
                params = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, min_p=0.0,
                                        repetition_penalty=1.0, presence_penalty=0.0,
                                        frequency_penalty=0.0, n=1, seed=0,
                                        max_tokens=maximum_output, ignore_eos=False,
                                        detokenize=True, skip_special_tokens=True)
                batch.append(row)
                requests.append(request)
                expected_ids.append(ids)
                prompt_stats.append(stats)
                parameters.append(params)
            except Exception as error:
                status["errors"] += 1
                save_error(row, args, identity, error)
        if not batch:
            atomic_json(status_path, status)
            continue
        started = time.monotonic()
        try:
            outputs = engine.generate(requests, parameters, use_tqdm=False)
            if len(outputs) != len(batch):
                raise RuntimeError("Engine returned the wrong number of request outputs")
        except Exception as error:
            for row, stats, params in zip(batch, prompt_stats, parameters):
                status["errors"] += 1
                save_error(row, args, identity, error, prompt_stats=stats,
                           max_tokens=params.max_tokens)
            status.update(state="engine_failed", finished_at=now())
            atomic_json(status_path, status)
            raise
        batch_seconds = time.monotonic() - started
        for row, ids, stats, params, output in zip(batch, expected_ids, prompt_stats,
                                                  parameters, outputs, strict=True):
            record = {**base_record(row, args, identity), "time": now(),
                      "prompt_stats": stats, "prompt_tokens": len(ids),
                      "max_model_len": maximum, "max_tokens": params.max_tokens,
                      "sampling_settings": settings, "batch_seconds": batch_seconds,
                      "batch_size": len(batch), "gpu": gpu}
            try:
                if len(output.outputs) != 1:
                    raise RuntimeError("Expected exactly one greedy completion")
                generated = output.outputs[0]
                actual_ids = list(output.prompt_token_ids or [])
                token_ids = list(generated.token_ids)
                record.update(output_text=generated.text, response=generated.text,
                              output_token_ids=token_ids, output_tokens=len(token_ids),
                              finish_reason=generated.finish_reason,
                              stop_reason=generated.stop_reason,
                              vllm_prompt_tokens=len(actual_ids),
                              vllm_prompt_ids_sha256=digest(actual_ids),
                              output_text_sha256=digest(generated.text.encode()),
                              output_token_ids_sha256=digest(token_ids),
                              generation_budget_exhausted=generated.finish_reason == "length",
                              context_limit_reached=(generated.finish_reason == "length" and
                                                     params.max_tokens == stats["context_remaining_tokens"]),
                              smoke_cap_reached=(generated.finish_reason == "length" and
                                                 params.max_tokens < stats["context_remaining_tokens"]))
                if actual_ids != ids:
                    raise RuntimeError("vLLM expanded prompt IDs differ from official AutoProcessor; "
                                       f"lengths {len(actual_ids)} != {len(ids)} or token IDs mismatch")
                if not output.finished or generated.finish_reason not in ("stop", "length"):
                    raise RuntimeError(f"Incomplete/invalid engine finish: {generated.finish_reason}")
                if len(token_ids) > params.max_tokens or len(ids) + len(token_ids) > maximum:
                    raise RuntimeError("Engine output exceeds the requested/model token budget")
                record["status"] = "completed"
                destination = args.output / "completed" / f"{row['dataset_index']:05d}.json"
                atomic_json(destination, record)
                status["completed"] += 1
                emit("task_completed", task_id=row["task_id"], prompt_tokens=len(ids),
                     output_tokens=len(token_ids), finish_reason=generated.finish_reason,
                     max_tokens=params.max_tokens, path=str(destination))
            except Exception as error:
                status["errors"] += 1
                save_error(row, args, identity, error, **record)
        status["updated_at"] = now()
        atomic_json(status_path, status)
    status.update(state="generation_complete" if not status["errors"] else "generation_incomplete",
                  finished_at=now())
    atomic_json(status_path, status)
    emit("worker_finished", **status)
    return 2 if status["errors"] else 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", "--max-num-seqs", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--smoke-limit", type=int)
    parser.add_argument("--smoke-max-tokens", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args(argv)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Require 0 <= shard-index < num-shards")
    if args.batch_size < 1 or args.max_num_batched_tokens < 1:
        parser.error("Batch sizes must be positive")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("GPU memory utilization must be between 0 and 1")
    if args.smoke_limit is not None and args.smoke_limit < 1:
        parser.error("Smoke limit must be positive")
    if args.smoke_max_tokens is not None:
        if args.smoke_max_tokens < 1 or args.smoke_limit is None:
            parser.error("An output cap is permitted only with --smoke-limit")
    args.model_path = args.model_path.expanduser().resolve()
    args.tasks = args.tasks.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    lock = args.output / "locks" / f"shard_{args.shard_index:03d}.lock"
    with exclusive_lock(lock):
        return run(args)


if __name__ == "__main__":
    sys.exit(main())
