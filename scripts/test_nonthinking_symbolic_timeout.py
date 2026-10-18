#!/usr/bin/env python3
"""Minimal cluster-side smoke test for the isolated SymPy equality guard."""

from run_qwen3_hard_eval_nonthinking import install_symbolic_equal_timeout


def main() -> None:
    install_symbolic_equal_timeout()
    from evalscope.metrics.math import parser as math_parser

    result = math_parser.symbolic_equal("2+2", "4")
    print(f"SYMBOLIC_TIMEOUT_SMOKE={result}", flush=True)
    if result is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
