"""Verify and publish the two completed 2B checkpoints without using a GPU."""

import json
from pathlib import Path

from verify_models import atomic_json, utc_now, verify_model


RUN = Path(__file__).resolve().parent


def main():
    plan = json.loads((RUN / "plan.json").read_text())
    state = json.loads((RUN / "status.json").read_text())
    records = []
    for model in plan["datasets"]:
        if model["name"] not in ("Qwen3-VL-2B-Instruct", "Qwen3-VL-2B-Thinking"):
            continue
        if state["models"][model["name"]]["state"] != "DOWNLOADED":
            raise RuntimeError("Download worker has not finished: " + model["name"])
        records.append(verify_model(model, Path(plan["root"]), Path(plan["publish_root"])))
        atomic_json(RUN / "TWO_B_READY.json", {"updated_at_utc": utc_now(), "models": records})


if __name__ == "__main__":
    main()
