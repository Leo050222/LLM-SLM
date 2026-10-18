"""Download pinned official models with the existing hf CLI, then verify and publish."""
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


RUN = Path(__file__).resolve().parent
HF = "/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/hf"
PYTHON = "/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python"
LOCK = threading.RLock()
STATE = {}


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def update(name, **values):
    with LOCK:
        STATE["models"][name].update(values)
        STATE["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_json(RUN / "status.json", STATE)


def measure(path, model):
    complete = 0
    total = 0
    for item in model["files"]:
        target = path / item["path"]
        if target.is_file() and target.stat().st_size == item["size"]:
            complete += 1
            total += item["size"]
    partial = 0
    cache = path / ".cache"
    if cache.is_dir():
        for item in cache.rglob("*.incomplete"):
            try:
                stat = item.stat()
            except FileNotFoundError:
                continue  # The Hub atomically renames a completed temporary file.
            partial += min(stat.st_size, stat.st_blocks * 512)
    return {"files_present": complete, "files_total": len(model["files"]),
            "complete_file_bytes": total,
            "partial_allocated_bytes": partial,
            "expected_bytes": sum(item["size"] for item in model["files"])}


def download_model(model, plan):
    name = model["name"]
    path = Path(plan["root"]) / name
    final = Path(plan["publish_root"]) / name
    if final.exists():
        source = final / "SOURCE.json"
        if source.is_file() and json.loads(source.read_text()).get("revision") == model["revision"]:
            update(name, state="ALREADY_PUBLISHED", path=str(final), **measure(final, model))
            return
        raise RuntimeError("Refusing to replace existing model: " + str(final))
    path.mkdir(parents=True, exist_ok=True)
    logfile = RUN / (name + ".log")
    # First use the Hub's native transfer engine. Later routes use ordinary HTTP
    # so a blocked Xet endpoint cannot prevent resumable downloads.
    routes = [("https://huggingface.co", True, False),
              ("https://huggingface.co", True, True),
              ("https://hf-mirror.com", True, True),
              ("https://hf-mirror.com", False, True)]
    for attempt, (endpoint, proxy, disable_xet) in enumerate(routes * 2, 1):
        env = os.environ.copy()
        env.update(HF_ENDPOINT=endpoint, HF_HUB_DISABLE_TELEMETRY="1",
                   HF_HUB_DISABLE_UPDATE_CHECK="1", HF_HUB_DOWNLOAD_TIMEOUT="60",
                   HF_HUB_ETAG_TIMEOUT="30", HF_HUB_DISABLE_PROGRESS_BARS="1",
                   HF_HUB_OFFLINE="0", HF_XET_CACHE=str(RUN / "xet_cache"))
        for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"]:
            env.pop(key, None)
        if proxy:
            env["HTTP_PROXY"] = env["HTTPS_PROXY"] = "http://127.0.0.1:7890"
        env["HF_HUB_DISABLE_XET"] = "1" if disable_xet else "0"
        cmd = [HF, "download", model["repo_id"], "--revision", model["revision"],
               "--local-dir", str(path), "--max-workers", "3", "--quiet"]
        update(name, state="DOWNLOADING", attempt=attempt, endpoint=endpoint,
               proxy=proxy, disable_xet=disable_xet, path=str(path))
        print("START", name, "attempt", attempt, endpoint, "xet", not disable_xet, flush=True)
        with logfile.open("ab", buffering=0) as output:
            process = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, env=env)
            update(name, pid=process.pid)
            last_bytes = -1
            last_change = time.monotonic()
            try:
                while process.poll() is None:
                    stats = measure(path, model)
                    present = stats["complete_file_bytes"] + stats["partial_allocated_bytes"]
                    # Xet may preallocate a destination and download out of order;
                    # actual log/temp-file activity also proves forward progress.
                    activity = logfile.stat().st_mtime_ns
                    for directory, pattern in [(path / ".cache", "*.incomplete"),
                                                (RUN / "xet_cache" / "logs", "*.log")]:
                        if directory.is_dir():
                            for active in directory.rglob(pattern):
                                try:
                                    activity = max(activity, active.stat().st_mtime_ns)
                                except FileNotFoundError:
                                    pass
                    signature = (present, activity)
                    if signature != last_bytes:
                        last_bytes = signature
                        last_change = time.monotonic()
                    update(name, **stats)
                    if time.monotonic() - last_change > 900:
                        process.terminate()
                        print("STALL_RETRY", name, flush=True)
                        break
                    time.sleep(5)
                try:
                    result = process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    result = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        stats = measure(path, model)
        if result == 0 and stats["files_present"] == stats["files_total"]:
            source = {key: value for key, value in model.items() if key != "files"}
            source.update(files=model["files"], transport="official Hugging Face CLI",
                          downloaded_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            atomic_json(path / "SOURCE.json", source)
            update(name, state="DOWNLOADED", **stats)
            print("DOWNLOADED", name, stats["expected_bytes"], flush=True)
            return
        update(name, state="RETRYING", last_exit_code=result, **stats)
        print("RETRY", name, "exit", result, flush=True)
    update(name, state="ERROR")
    raise RuntimeError(name + ": download failed; see " + str(logfile))


def main():
    lock = (RUN / ".run.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((RUN / "plan.json").read_text())
    STATE.update(state="RUNNING", models={model["name"]: {"state": "PENDING"}
                                         for model in plan["datasets"]})
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(download_model, model, plan): model["name"]
                   for model in plan["datasets"]}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as error:
                errors.append(str(error))
                update(futures[future], state="ERROR", error=str(error))
                print("ERROR", str(error), flush=True)
    if errors:
        STATE.update(state="ERROR", errors=errors)
        atomic_json(RUN / "status.json", STATE)
        return 1
    STATE["state"] = "VERIFYING"
    atomic_json(RUN / "status.json", STATE)
    # The verifier is deployed separately while weights are transferring.
    deadline = time.monotonic() + 180
    while not (RUN / "verify_models.py").is_file():
        if time.monotonic() > deadline:
            raise RuntimeError("Verifier was not deployed")
        time.sleep(5)
    result = subprocess.run([PYTHON, "-u", str(RUN / "verify_models.py"),
                             "--plan", str(RUN / "plan.json")])
    STATE["state"] = "COMPLETE" if result.returncode == 0 else "ERROR"
    STATE["verification_exit_code"] = result.returncode
    atomic_json(RUN / "status.json", STATE)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
