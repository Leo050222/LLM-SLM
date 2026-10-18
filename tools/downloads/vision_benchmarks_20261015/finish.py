"""Finish the current download with offline verification and a dataset index."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
PYTHON = "/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python"


def main():
    lock = (ROOT / ".finish.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("FINISH_ALREADY_RUNNING", flush=True)
        return 0
    deadline = time.monotonic() + 3 * 60 * 60
    while time.monotonic() < deadline:
        path = ROOT / "status.json"
        if path.exists():
            status = json.loads(path.read_text())
            if status.get("state") == "COMPLETE":
                break
            if status.get("state") == "ERROR":
                print("DOWNLOAD_ERROR: inspect status.json", flush=True)
                return 1
        time.sleep(10)
    else:
        print("DOWNLOAD_WAIT_TIMEOUT", flush=True)
        return 1
    print("START_OFFLINE_VERIFICATION", flush=True)
    subprocess.run([PYTHON, str(ROOT / "verify.py"), "--root", str(ROOT)], check=True)
    subprocess.run([PYTHON, str(ROOT / "make_readme.py"), "--root", str(ROOT)], check=True)
    report = json.loads((ROOT / "VERIFIED.json").read_text())
    assert report["ok"] and len(report["datasets"]) == 8
    marker = {"ok": True, "root": str(ROOT), "datasets": 8,
              "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "verified_report": str(ROOT / "VERIFIED.json")}
    temporary = ROOT / "COMPLETE.json.tmp"
    temporary.write_text(json.dumps(marker, indent=2) + "\n")
    os.replace(temporary, ROOT / "COMPLETE.json")
    print("ALL_EIGHT_BENCHMARKS_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
