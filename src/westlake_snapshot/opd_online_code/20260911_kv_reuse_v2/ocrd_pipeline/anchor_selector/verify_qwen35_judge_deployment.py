#!/usr/bin/env python3
"""Read-only acceptance checks for the local Qwen3.5-27B judge deployment."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR.parent))

from anchor_selector.openlux_client import SYSTEM_PROMPT_SHA256


ROOT = Path("/storage/yukaichengLab/wangxin/test_eiloeo")
INPUT = ROOT / "ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2"


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected object in {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    spec = load_json(MODULE_DIR / "env-qwen35-27b-judge.json")
    spec_hash = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:8]
    require(spec_hash == "b1bfb1e9", f"environment spec drift: {spec_hash}")

    config = load_json(MODULE_DIR / "config_qwen35_27b_local_8gpu.json")
    require(len(config["api_urls"]) == 8, "formal config must expose eight endpoints")
    require(config["judge_max_tokens"] == 2048, "formal max tokens must be 2048")
    require(
        config["judge_extra_body"]["chat_template_kwargs"]["enable_thinking"] is False,
        "judge must use native non-thinking output for strict JSON",
    )

    smoke = load_json(INPUT / "judge_local_qwen35_27b_smoke20_v2" / "summary.json")
    require(smoke["status"] == "BATCH_JUDGE_COMPLETE", "smoke is incomplete")
    require(smoke["completed_traces"] == 20, "smoke trace count mismatch")
    require(smoke["failed_traces"] == 0 and not smoke["missing_traces"], "smoke failures")

    quality = INPUT / "judge_local_qwen35_27b_calibration_careful_nonthinking_v1"
    dev = load_json(quality / "dev_natural_200_seed20260826" / "summary.json")
    balanced = load_json(quality / "heldout_balanced_200_seed20260827" / "summary.json")
    for name, summary in (("dev", dev), ("balanced", balanced)):
        require(summary["status"] == "CALIBRATION_COMPLETE", f"{name} incomplete")
        require(summary["failed_samples"] == 0, f"{name} contains failed samples")
    require(dev["select_anchor_precision"] >= 0.45, "dev precision gate failed")
    require(dev["select_anchor_recall"] >= 0.60, "dev recall gate failed")
    require(balanced["select_anchor_f1"] >= 0.72, "balanced F1 gate failed")
    require(balanced["select_anchor_recall"] >= 0.70, "balanced recall gate failed")
    natural = load_json(quality / "heldout_natural_500_seed20260828" / "summary.json")
    require(natural["status"] == "CALIBRATION_COMPLETE", "natural incomplete")
    require(natural["failed_samples"] == 0, "natural contains failed samples")
    require(natural["select_anchor_precision"] >= 0.45, "natural precision gate failed")
    require(natural["select_anchor_recall"] >= 0.60, "natural recall gate failed")

    script = (MODULE_DIR / "submit_qwen35_27b_judge_8gpu.slurm").read_text(
        encoding="utf-8"
    )
    for expected in (
        "#SBATCH --nodes=1",
        "#SBATCH --gres=gpu:H800:8",
        "#SBATCH --exclusive",
        "--max-model-len 49152",
        "--workers 48",
        "--candidate-batch-size 4",
    ):
        require(expected in script, f"formal script missing: {expected}")

    records = list(INPUT.glob("shards/shard_*/candidate_records/trace_*.json"))
    require(len(records) == 14158, f"candidate record count mismatch: {len(records)}")
    print(
        json.dumps(
            {
                "status": "QWEN35_JUDGE_DEPLOYMENT_OK",
                "environment": f"qwen35-27b-anchor-judge@{spec_hash}",
                "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
                "candidate_records": len(records),
                "smoke_judge_calls": smoke["judge_calls"],
                "heldout_balanced_f1": balanced["select_anchor_f1"],
                "heldout_natural_precision": natural["select_anchor_precision"],
                "heldout_natural_recall": natural["select_anchor_recall"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
