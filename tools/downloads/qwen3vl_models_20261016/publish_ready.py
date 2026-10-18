"""Publish the priority checkpoint after its existing downloader completes.

The download worker retains ownership until it has written SOURCE.json and marked
the model DOWNLOADED. Verification is offline and never allocates a GPU.
"""

import fcntl
import json
from pathlib import Path
import time

from verify_models import atomic_json, utc_now, verify_model


RUN = Path(__file__).resolve().parent
NAME = "Qwen3-VL-2B-Instruct"


def main():
    lock = (RUN / ".publish_ready.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((RUN / "plan.json").read_text())
    model = next(item for item in plan["datasets"] if item["name"] == NAME)
    stage = Path(plan["root"])
    final = Path(plan["publish_root"]) / NAME
    while True:
        if (final / "VERIFIED.json").is_file():
            print("ALREADY_VERIFIED", str(final), flush=True)
            return
        status = json.loads((RUN / "status.json").read_text())
        # If the all-model verification already started, it owns publication.
        if status["state"] in ("VERIFYING", "COMPLETE"):
            print("OWNED_BY_MAIN_VERIFIER", status["state"], flush=True)
            return
        if (status.get("models", {}).get(NAME, {}).get("state") == "DOWNLOADED"
                and (stage / NAME / "SOURCE.json").is_file()):
            record = verify_model(model, stage, Path(plan["publish_root"]))
            atomic_json(RUN / "PRIORITY_READY.json", {
                "state": "VERIFIED", "published_at_utc": utc_now(), "model": record})
            return
        time.sleep(10)


if __name__ == "__main__":
    main()
