"""Checkpointed, read-only search of all accessible BOS object names."""
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import pathlib
import subprocess
import threading
import time
from zoneinfo import ZoneInfo

from check_bos_prefix import (
    BceCredentials, BceClientConfiguration, BosClient, HTTPS, NoRetryPolicy,
    safe_error, safe_scalar,
)

ROOT = pathlib.Path(__file__).parent / "wangxin-full-search-20261011"
ROOT.mkdir(exist_ok=True)
LOCK = threading.RLock()
STOP = threading.Event()


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main():
    response = subprocess.run([
        "kubectl", "--kubeconfig", "/Users/leolee/Downloads/auwomo-h800-team",
        "--context", "cluster1-team-auwomo", "--request-timeout=15s",
        "-n", "dcloud-auwomo", "get", "secret",
        "jfs-private-prod-awomo-self-hosted-secret", "-o", "json",
    ], capture_output=True, text=True, check=True)
    secret = json.loads(response.stdout)["data"]
    credentials = BceCredentials(base64.b64decode(secret["access-key"]),
                                 base64.b64decode(secret["secret-key"]))
    initial_client = BosClient(BceClientConfiguration(
        credentials=credentials, endpoint="https://bj.bcebos.com", protocol=HTTPS,
        connection_timeout_in_mills=10000, retry_policy=NoRetryPolicy(),
    ))
    buckets = [{"name": safe_scalar(b.name), "region": safe_scalar(b.location)}
               for b in initial_client.list_buckets().buckets]
    states = {}
    for bucket in buckets:
        path = ROOT / (bucket["name"] + ".json")
        if path.exists():
            state = json.loads(path.read_text())
            if state["status"] == "error":
                state["status"] = "pending"
                state.pop("error", None)
        else:
            state = {"bucket": bucket["name"], "region": bucket["region"],
                     "status": "pending", "objects_scanned": 0, "pages_scanned": 0,
                     "matches": 0, "next_marker": None, "started_at": now()}
        states[bucket["name"]] = state

    def snapshot():
        with LOCK:
            rows = [dict(states[k]) for k in sorted(states)]
            report = {"updated_at": now(), "read_only": True,
                      "search": "case-insensitive substring wangxin in all object keys",
                      "file_contents_scanned": False,
                      "completed_buckets": sum(r["status"] == "complete" for r in rows),
                      "total_buckets": len(rows),
                      "objects_scanned": sum(r["objects_scanned"] for r in rows),
                      "matches": sum(r["matches"] for r in rows), "buckets": rows}
            save_json(ROOT / "summary.json", report)
        return report

    def progress():
        while not STOP.wait(25):
            report = snapshot()
            print(json.dumps({k: report[k] for k in ["updated_at", "completed_buckets",
                  "total_buckets", "objects_scanned", "matches"]}), flush=True)

    def scan(bucket):
        name = bucket["name"]
        with LOCK:
            state = states[name]
            if state["status"] == "complete":
                return
            state["status"] = "running"
        client = BosClient(BceClientConfiguration(
            credentials=credentials, endpoint="https://" + bucket["region"] + ".bcebos.com",
            protocol=HTTPS, connection_timeout_in_mills=12000, retry_policy=NoRetryPolicy(),
        ))
        errors = 0
        matches_path = ROOT / (name + "-matches.jsonl")
        try:
            with matches_path.open("a") as matches_file:
                while True:
                    marker = state["next_marker"]
                    try:
                        response = client.list_objects(name, max_keys=1000, marker=marker)
                    except Exception as error:
                        errors += 1
                        if errors < 3 and getattr(error, "status_code", None) not in [401, 403, 404]:
                            time.sleep(1)
                            continue
                        raise
                    errors = 0
                    directories = getattr(response, "common_prefixes", None) or []
                    if directories:
                        raise RuntimeError("Directory-only response requires hierarchical traversal")
                    objects = getattr(response, "contents", None) or []
                    matches = []
                    for obj in objects:
                        key = safe_scalar(obj.key)
                        if "wangxin" in key.casefold():
                            entry = {"bucket": name, "key": key, "size_bytes": obj.size,
                                     "last_modified": safe_scalar(getattr(obj, "last_modified", None))}
                            matches.append(entry)
                            matches_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    matches_file.flush()
                    truncated = bool(getattr(response, "is_truncated", False))
                    next_marker = safe_scalar(getattr(response, "next_marker", None))
                    if truncated and not next_marker and objects:
                        next_marker = safe_scalar(objects[-1].key)
                    with LOCK:
                        state["objects_scanned"] += len(objects)
                        state["pages_scanned"] += 1
                        state["matches"] += len(matches)
                        state["next_marker"] = next_marker
                        state["updated_at"] = now()
                        if not truncated:
                            state["status"] = "complete"
                        save_json(ROOT / (name + ".json"), state)
                    if matches:
                        print(json.dumps({"event": "matches", "bucket": name,
                                          "new_matches": len(matches), "sample": matches[:3]},
                                         ensure_ascii=False), flush=True)
                    if not truncated:
                        print(json.dumps({"event": "bucket_complete", "bucket": name,
                                          "objects_scanned": state["objects_scanned"],
                                          "matches": state["matches"]}), flush=True)
                        break
                    if not next_marker or next_marker == marker:
                        raise RuntimeError("Pagination marker did not advance")
        except Exception as error:
            with LOCK:
                state["status"] = "error"
                state["error"] = safe_error(error)
                state["updated_at"] = now()
                save_json(ROOT / (name + ".json"), state)
            print(json.dumps({"event": "bucket_error", "bucket": name,
                              "error": state["error"]}), flush=True)

    reporter = threading.Thread(target=progress, daemon=True)
    reporter.start()
    snapshot()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(scan, sorted(buckets, key=lambda b: b["name"])))
    STOP.set()
    reporter.join()
    report = snapshot()
    print(json.dumps({"event": "search_finished", "completed_buckets": report["completed_buckets"],
                      "total_buckets": report["total_buckets"],
                      "objects_scanned": report["objects_scanned"], "matches": report["matches"]}), flush=True)
    print("LOCAL_REPORT=" + str(ROOT / "summary.json"), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"fatal_error": safe_error(error)}), flush=True)
        raise SystemExit(1)
