#!/usr/bin/env python3
"""Launch vLLM while skipping its incompatible Prometheus middleware.

The validated verl-opd environment combines vLLM 0.23.0, FastAPI 0.137.0,
Starlette 1.3.1, and prometheus-fastapi-instrumentator 8.0.0. The metrics
middleware assumes every Starlette route has a ``path`` attribute, which is no
longer true for ``_IncludedRouter``. The model API itself is unaffected, so the
evaluation server omits only Prometheus instrumentation.
"""

import runpy

from vllm.entrypoints.serve.instrumentator import metrics


def _skip_metrics_router(_app):
    return None


def main():
    metrics.attach_router = _skip_metrics_router
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
