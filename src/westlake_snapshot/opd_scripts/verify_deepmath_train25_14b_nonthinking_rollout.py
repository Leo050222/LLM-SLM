#!/usr/bin/env python3
"""Run the strict train-25% rollout verifier for frozen Qwen3-14B."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import verify_deepmath_train25_nonthinking_rollout as verifier


EXPECTED_MODEL = "/storage/yukaichengLab/share/model/Qwen3-14B"
EXPECTED_TOKENIZER = "/storage/yukaichengLab/share/model/Qwen3-1.7B"


def main() -> None:
    verifier.EXPECTED_MODEL = EXPECTED_MODEL
    verifier.EXPECTED_TOKENIZER = EXPECTED_TOKENIZER
    verifier.main()

    output_root = Path(sys.argv[sys.argv.index("--output-root") + 1])
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("model") != EXPECTED_MODEL:
        raise RuntimeError("Verified summary model mismatch")
    if summary.get("tokenizer") != EXPECTED_TOKENIZER:
        raise RuntimeError("Verified summary tokenizer mismatch")
    print("DEEPMATH_TRAIN25_QWEN3_14B_NATIVE_NONTHINKING_ROLLOUT_OK")


if __name__ == "__main__":
    main()
