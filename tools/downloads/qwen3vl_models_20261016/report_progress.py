"""Report actual checkpoint bytes, excluding obsolete Xet partial files."""

import datetime
import json
import os
from pathlib import Path


RUN = Path(__file__).resolve().parent


def process_started_at(pid):
    try:
        fields = Path("/proc/%d/stat" % pid).read_text().rsplit(")", 1)[1].split()
        ticks = int(fields[19])
        boot = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
                    if line.startswith("btime "))
        return boot + ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, StopIteration):
        return float("inf")


def main():
    plan = json.loads((RUN / "plan.json").read_text())
    state = json.loads((RUN / "status.json").read_text())
    result = {"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "main_state": state["state"], "models": []}
    for model in plan["datasets"]:
        path = Path(plan["root"]) / model["name"]
        final = Path(plan["publish_root"]) / model["name"]
        info = state.get("models", {}).get(model["name"], {})
        current_started = process_started_at(info.get("pid", 0))
        details = []
        for item in model["files"]:
            if not item["path"].endswith(".safetensors"):
                continue
            count, source = 0, "pending"
            for parent, label in [(final, "published"), (path, "complete")]:
                target = parent / item["path"]
                if target.is_file():
                    count, source = target.stat().st_size, label
                    break
            else:
                active = []
                pattern = "*.%s.*.incomplete" % item["sha256"]
                for partial in (path / ".cache/huggingface/download").glob(pattern):
                    try:
                        stat = partial.stat()
                    except FileNotFoundError:
                        continue
                    if stat.st_mtime >= current_started:
                        active.append(stat)
                if active:
                    latest = max(active, key=lambda stat: stat.st_mtime_ns)
                    count, source = min(latest.st_size, latest.st_blocks * 512), "active_partial"
            details.append({"path": item["path"], "bytes": min(count, item["size"]),
                            "total_bytes": item["size"], "source": source})
        downloaded = sum(item["bytes"] for item in details)
        total = sum(item["total_bytes"] for item in details)
        result["models"].append({"name": model["name"], "state": info.get("state"),
                                 "bytes": downloaded, "total_bytes": total,
                                 "percent": round(downloaded / total * 100, 2),
                                 "verified": (final / "VERIFIED.json").is_file(),
                                 "shards": details})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
