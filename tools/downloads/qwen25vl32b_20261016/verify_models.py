"""Verify pinned local Qwen2.5-VL checkpoints and publish without replacement.

Usage: python -u verify_models.py --plan /path/to/plan.json
No model parameters are materialized, no GPU is used, and no network is needed.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import tempfile
import time
import traceback


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix="." + path.name + ".", delete=False) as out:
        temporary = Path(out.name)
        try:
            json.dump(value, out, ensure_ascii=False, indent=2)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def relative_path(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Expected a nonempty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("Unsafe relative path: " + value)
    return path


def expected_files(model):
    result = {}
    for entry in model["files"]:
        path = str(relative_path(entry["path"]))
        if path in result:
            raise ValueError("Duplicate planned file: " + path)
        if not isinstance(entry["size"], int) or entry["size"] < 0:
            raise ValueError("Invalid planned size: " + path)
        result[path] = entry
    if not result:
        raise ValueError("Empty model file plan")
    return result


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_for(path, model):
    source_path = path / "SOURCE.json"
    if not source_path.is_file():
        raise ValueError("Missing completed download provenance: " + str(source_path))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    for field in ("repo_id", "revision"):
        if source.get(field) != model[field]:
            raise ValueError("SOURCE.json " + field + " differs from pinned plan: " + str(path))
    return source


def link_reserved_publication(source, destination):
    """Publish on filesystems without RENAME_NOREPLACE, never replacing a model.

    mkdir exclusively reserves a real destination directory. link(2) then exposes
    regular files without copying or overwriting anything. The readiness record
    is linked last. A recovery marker and the staged originals survive a partial
    attempt; readers must wait for VERIFIED.json. Download caches stay staged.
    """
    marker_name = ".PUBLICATION_IN_PROGRESS.json"
    identity = {"source": str(source), "destination": str(destination),
                "method": "exclusive-mkdir-and-no-replace-hardlinks"}
    staged_marker = source / marker_name
    if staged_marker.is_file():
        if json.loads(staged_marker.read_text()) != identity:
            raise RuntimeError("Staged publication marker differs from this attempt")
    else:
        atomic_json(staged_marker, identity)
    try:
        destination.mkdir()
    except FileExistsError:
        marker = destination / marker_name
        if (destination.is_symlink() or not marker.is_file()
                or json.loads(marker.read_text()) != identity):
            raise FileExistsError("Refusing to replace existing model: " + str(destination))
    else:
        # This is an exclusive create; a conflicting entry can never be replaced.
        os.link(staged_marker, destination / marker_name)
    files = sorted(item for item in source.iterdir()
                   if item.name not in (marker_name, "VERIFIED.json")
                   and item.is_file() and not item.is_symlink())
    for item in files:
        target = destination / item.name
        try:
            os.link(item, target)
        except FileExistsError:
            if target.is_symlink() or not os.path.samefile(item, target):
                raise FileExistsError("Refusing to replace publication entry: " + str(target))
    ready = destination / "VERIFIED.json"
    try:
        os.link(source / "VERIFIED.json", ready)
    except FileExistsError:
        current = json.loads(ready.read_text())
        expected = json.loads((source / "VERIFIED.json").read_text())
        if (ready.is_symlink() or any(current.get(field) != expected.get(field)
                for field in ("state", "name", "repo_id", "revision", "path", "bytes", "files"))):
            raise FileExistsError("Refusing to replace different verification record: " + str(ready))
    atomic_json(destination / "PUBLICATION.json", {**identity, "state": "COMPLETE",
                                                 "completed_at_utc": utc_now(),
                                                 "directory_atomic": False,
                                                 "readiness_record": "VERIFIED.json"})
    (destination / marker_name).unlink()
    # Retain staged files and its marker as recovery evidence until the full
    # run is complete. Hardlinks do not allocate a second weight copy.
    print("PUBLICATION_FALLBACK", str(destination), identity["method"], flush=True)


def rename_no_replace(source, destination):
    """Prefer atomic publication, with an exclusive reserved-directory fallback."""
    if sys.platform != "linux":
        raise RuntimeError("Atomic publication requires Linux renameat2(RENAME_NOREPLACE)")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        return link_reserved_publication(source, destination)
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result:
        code = ctypes.get_errno()
        if code in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError("Refusing to replace existing model: " + str(destination))
        if code in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
            return link_reserved_publication(source, destination)
        raise OSError(code, os.strerror(code), str(source), None, str(destination))


def verify_config(config):
    if config.get("model_type") != "qwen2_5_vl":
        raise ValueError("config.json model_type must be qwen2_5_vl")
    if "Qwen2_5_VLForConditionalGeneration" not in config.get("architectures", []):
        raise ValueError("config.json lacks the Qwen2.5-VL generation architecture")
    if not isinstance(config.get("vision_config"), dict):
        raise ValueError("config.json lacks vision_config")
    # Official Qwen2.5-VL checkpoints use the original flattened text fields;
    # recent Transformers versions may serialize a nested text_config instead.
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        raise ValueError("config.json has invalid text_config")
    for field in ("hidden_size", "intermediate_size", "num_attention_heads",
                  "num_hidden_layers", "num_key_value_heads", "vocab_size"):
        value = text_config.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("config.json lacks positive integer text field " + field)
    dtype = config.get("dtype", config.get("torch_dtype"))
    if dtype is None:
        dtype = text_config.get("dtype", text_config.get("torch_dtype"))
    if dtype != "bfloat16":
        raise ValueError("Expected the official BF16 checkpoint config, got " + repr(dtype))
    return text_config


def verify_structure(path, planned, config):
    from safetensors import safe_open

    weights = {name for name in planned if name.endswith(".safetensors")}
    if not weights:
        raise ValueError("No safetensors weights in plan")
    index_names = [name for name in planned if name.endswith(".safetensors.index.json")]
    if len(index_names) > 1:
        raise ValueError("Multiple safetensors indexes in plan")
    weight_map = None
    if index_names:
        index = json.loads((path / index_names[0]).read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Missing or empty index weight_map")
        if not all(isinstance(key, str) and isinstance(value, str)
                   for key, value in weight_map.items()):
            raise ValueError("Invalid index weight_map entries")
        indexed_shards = {str(relative_path(name)) for name in weight_map.values()}
        if indexed_shards != weights:
            raise ValueError("Index shards do not exactly match planned safetensors: "
                             + repr({"indexed": sorted(indexed_shards), "planned": sorted(weights)}))
    elif len(weights) != 1 or "model.safetensors" not in weights:
        raise ValueError("Multiple or nonstandard safetensors require an index")

    keys_by_shard = {}
    tensor_count = 0
    for name in sorted(weights):
        # safe_open maps the file and reads its header; get_tensor is never called.
        with safe_open(str(path / name), framework="pt", device="cpu") as checkpoint:
            keys = set(checkpoint.keys())
        if not keys:
            raise ValueError("Empty safetensors shard: " + name)
        if weight_map is not None:
            expected = {key for key, shard in weight_map.items() if shard == name}
            if keys != expected:
                raise ValueError("Safetensors/index key mismatch in " + name + ": "
                                 + repr({"missing": sorted(expected - keys)[:10],
                                         "unindexed": sorted(keys - expected)[:10]}))
        keys_by_shard[name] = len(keys)
        tensor_count += len(keys)
    verify_config(config)
    return {"shards": len(weights), "tensors": tensor_count,
            "keys_per_shard": keys_by_shard, "index_present": bool(index_names)}


def verify_processor(path, config):
    # Enforce offline operation even if the caller did not configure its environment.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from PIL import Image
    import torch
    import transformers
    import safetensors
    from transformers import AutoConfig, AutoProcessor

    loaded_config = AutoConfig.from_pretrained(str(path), local_files_only=True,
                                               trust_remote_code=False)
    if loaded_config.model_type != "qwen2_5_vl":
        raise ValueError("AutoConfig returned the wrong model type")
    processor = AutoProcessor.from_pretrained(str(path), local_files_only=True,
                                              trust_remote_code=False)
    picture = Image.new("RGB", (64, 64), (36, 120, 200))
    messages = [{"role": "user", "content": [
        {"type": "image", "image": picture},
        {"type": "text", "text": "What color is this square?"}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True,
                                            tokenize=True, return_dict=True,
                                            return_tensors="pt")
    required = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
    for key in required:
        if key not in inputs or not isinstance(inputs[key], torch.Tensor) or inputs[key].numel() == 0:
            raise ValueError("Processor smoke test lacks nonempty tensor " + key)
        if inputs[key].device.type != "cpu":
            raise ValueError("Processor smoke test unexpectedly used a non-CPU device")
    image_token_id = config.get("image_token_id", config.get("text_config", {}).get("image_token_id"))
    if not isinstance(image_token_id, int):
        raise ValueError("config.json lacks integer image_token_id")
    image_token_count = int((inputs["input_ids"] == image_token_id).sum().item())
    merge_size = int(processor.image_processor.merge_size)
    expected_tokens = sum(int(row.prod().item()) // (merge_size * merge_size)
                          for row in inputs["image_grid_thw"])
    if not image_token_count or image_token_count != expected_tokens:
        raise ValueError("Image token/grid mismatch: " + repr((image_token_count, expected_tokens)))
    if inputs["image_grid_thw"].shape != (1, 3):
        raise ValueError("Processor smoke test expected exactly one image")
    return {"config_class": type(loaded_config).__name__,
            "processor_class": type(processor).__name__, "image_size": [64, 64],
            "image_tokens": image_token_count,
            "tensor_shapes": {key: list(inputs[key].shape) for key in required},
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "transformers": transformers.__version__,
                         "safetensors": safetensors.__version__},
            "offline": True, "device": "cpu"}


def verify_model(model, staging_root, publish_root):
    # Serialize this run's verifier and priority publisher for each checkpoint.
    name = str(relative_path(model["name"]))
    if len(PurePosixPath(name).parts) != 1:
        raise ValueError("Model name must be one directory component")
    with (staging_root / (".verify_" + name + ".lock")).open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return verify_model_locked(model, staging_root, publish_root)


def verify_model_locked(model, staging_root, publish_root):
    name = str(relative_path(model["name"]))
    if len(PurePosixPath(name).parts) != 1:
        raise ValueError("Model name must be one directory component")
    final = publish_root / name
    # lexists also rejects dangling destination symlinks rather than replacing them.
    marker = final / ".PUBLICATION_IN_PROGRESS.json"
    recovering = (marker.is_file() and not final.is_symlink()
                  and json.loads(marker.read_text()).get("source") == str(staging_root / name))
    reuse = os.path.lexists(final) and not recovering
    path = final if reuse else staging_root / name
    if path.is_symlink() or not path.is_dir():
        raise ValueError("Expected a real model directory: " + str(path))
    source = source_for(path, model)
    planned = expected_files(model)
    source_hashes = {entry["path"]: entry.get("sha256") for entry in source.get("files", [])
                     if isinstance(entry, dict) and isinstance(entry.get("path"), str)}
    checked = []
    parsed_json = {}
    print("VERIFY_START", name, "reuse" if reuse else "staged", flush=True)
    for name_in_model, entry in planned.items():
        file_path = path / name_in_model
        if file_path.is_symlink() or not file_path.is_file():
            raise ValueError("Missing or symlinked expected file: " + str(file_path))
        if not file_path.resolve().is_relative_to(path.resolve()):
            raise ValueError("Expected file escapes model directory: " + str(file_path))
        stat_before = file_path.stat()
        if stat_before.st_size != entry["size"]:
            raise ValueError("Size mismatch: " + str(file_path) + ": "
                             + repr((stat_before.st_size, entry["size"])))
        actual_hash = sha256_file(file_path)
        stat_after = file_path.stat()
        if (stat_before.st_size, stat_before.st_mtime_ns, stat_before.st_ino) != (
                stat_after.st_size, stat_after.st_mtime_ns, stat_after.st_ino):
            raise ValueError("File changed during verification: " + str(file_path))
        expected_hash = entry.get("sha256") or source_hashes.get(name_in_model)
        if name_in_model.endswith(".safetensors") and not expected_hash:
            raise ValueError("No trusted download SHA256 for weight: " + name_in_model)
        if expected_hash and actual_hash.lower() != expected_hash.lower():
            raise ValueError("SHA256 mismatch: " + str(file_path))
        if name_in_model.endswith(".json"):
            parsed_json[name_in_model] = json.loads(file_path.read_text(encoding="utf-8"))
        checked.append({"path": name_in_model, "size": stat_after.st_size,
                        "sha256": actual_hash, "expected_sha256_matched": bool(expected_hash)})
        print("FILE_OK", name, name_in_model, stat_after.st_size, actual_hash, flush=True)
    config = parsed_json.get("config.json")
    if not isinstance(config, dict):
        raise ValueError("Missing parsed config.json")
    structure = verify_structure(path, planned, config)
    print("HEADERS_OK", name, structure["shards"], structure["tensors"], flush=True)
    processor = verify_processor(path, config)
    print("PROCESSOR_OK", name, processor["image_tokens"], flush=True)
    record = {"state": "VERIFIED", "name": name, "repo_id": model["repo_id"],
              "revision": model["revision"], "path": str(final),
              "verified_at_utc": utc_now(), "reused_existing": reuse,
              "bytes": sum(entry["size"] for entry in planned.values()),
              "files": checked, "safetensors": structure, "processor": processor}
    # The readiness record travels with an atomic rename where supported, and is
    # exposed last by the reserved-directory fallback on this shared filesystem.
    atomic_json(path / "VERIFIED.json", record)
    if not reuse:
        if recovering:
            link_reserved_publication(path, final)
        else:
            rename_no_replace(path, final)
    print("PUBLISHED" if not reuse else "REUSED", name, str(final),
          "bytes", record["bytes"], "shards", structure["shards"], flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    staging_root = Path(plan["root"]).resolve()
    publish_root = Path(plan["publish_root"]).resolve()
    if staging_root == publish_root:
        raise ValueError("Staging and publication roots must be different")
    if not publish_root.is_dir():
        raise ValueError("Publication root does not exist")
    models = plan["datasets"]
    if not models or len({model["name"] for model in models}) != len(models):
        raise ValueError("Empty or duplicate model plan")
    records = []
    errors = []
    for model in models:
        try:
            records.append(verify_model(model, staging_root, publish_root))
        except Exception as error:
            failure = {"name": model.get("name"), "error": type(error).__name__ + ": " + str(error)}
            errors.append(failure)
            print("VERIFICATION_FAILED", json.dumps(failure, ensure_ascii=False),
                  file=sys.stderr, flush=True)
            traceback.print_exc()
    if errors:
        print("FAILED", json.dumps(errors, ensure_ascii=False), file=sys.stderr, flush=True)
        return 1
    summary = {"state": "COMPLETE", "verified_at_utc": utc_now(),
               "plan": str(plan_path), "publish_root": str(publish_root),
               "models": records, "model_count": len(records),
               "total_bytes": sum(record["bytes"] for record in records)}
    atomic_json(plan_path.parent / "VERIFIED.json", summary)
    atomic_json(plan_path.parent / "COMPLETE.json", summary)
    print("COMPLETE", len(records), "models", summary["total_bytes"], "bytes", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
