#!/usr/bin/env python3
"""Non-launching integrity checks for the DeepMath exact-RKL formal run."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--preflight-run-dir", type=Path, required=True)
    parser.add_argument("--smoke-summary", type=Path, required=True)
    args = parser.parse_args()

    smoke = json.loads(args.smoke_summary.read_text(encoding="utf-8"))
    assert smoke["status"] == "DEEPMATH_FULL_VOCAB_RKL_MAXLEN_SMOKE_OK"
    assert smoke["objective"] == "exact_full_vocabulary_reverse_kl_student_to_teacher"
    assert smoke["model_vocab_size"] == 151936
    assert smoke["tokenizer_vocab_size"] == 151669
    assert smoke["examples"] == 4
    assert smoke["total_response_tokens"] == 36016
    assert all(row["student_peak_allocated_gib"] < 80 for row in smoke["rows"])
    assert all(row["teacher_peak_allocated_gib"] < 80 for row in smoke["rows"])
    assert all(row["mean_reverse_kl"] > 0 for row in smoke["rows"])

    spec = importlib.util.spec_from_file_location("formal_full_rkl", args.trainer)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.TOTAL_STEPS == 885
    assert module.TRAIN_ROWS == 56640
    assert module.GLOBAL_BATCH_SIZE == 64
    assert module.LOCAL_BATCH_SIZE == 16
    assert module.STUDENT_WORLD_SIZE == 4
    assert module.SAVE_STEPS == (177, 354, 531, 708, 885)
    assert module.MODEL_VOCAB_SIZE == 151936
    assert module.TOKENIZER_VOCAB_SIZE == 151669
    assert module.MAX_MODEL_LEN == 40960
    assert module.MAX_RESPONSE_TOKENS == 40264
    assert module.CHUNK_TOKENS == 256

    module.phase_init(
        argparse.Namespace(run_dir=args.preflight_run_dir)
    )
    all_indices: list[int] = []
    for step in range(1, module.TOTAL_STEPS + 1):
        for rank in range(module.STUDENT_WORLD_SIZE):
            batch = module.batch_indices(args.preflight_run_dir, step, rank)
            assert len(batch) == module.LOCAL_BATCH_SIZE
            all_indices.extend(batch)
    assert len(all_indices) == module.TRAIN_ROWS
    assert sorted(all_indices) == list(range(module.TRAIN_ROWS))

    run_manifest = json.loads(
        (args.preflight_run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["objective"] == "exact_full_model_vocabulary_kl_student_to_teacher"
    assert run_manifest["save_steps"] == [177, 354, 531, 708, 885]
    print(
        "FULL_VOCAB_RKL_FORMAL_PREFLIGHT_OK",
        json.dumps(
            {
                "smoke_mean_reverse_kl": smoke["mean_reverse_kl_token_weighted"],
                "train_rows": len(all_indices),
                "steps": module.TOTAL_STEPS,
                "save_steps": list(module.SAVE_STEPS),
                "order_sha256": run_manifest["order_sha256"],
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
