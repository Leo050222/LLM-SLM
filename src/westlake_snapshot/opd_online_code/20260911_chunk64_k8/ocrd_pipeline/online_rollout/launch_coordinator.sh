#!/bin/bash
# Run on a networked node with the shared filesystem mounted.
set -euo pipefail
: "${OPD_CODE_ROOT:?}"
: "${OPD_ONLINE_RUN:?}"
: "${OPD_PYTHON:?}"
: "${OPD_JUDGE_CONFIG:?}"
: "${OPD_JUDGE_SECRETS:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
# A finished/failed coordinator releases its resident workers.
trap 'touch "$OPD_ONLINE_RUN/queue/STOP"' EXIT
for ((i=0; i<180; i++)); do
  if [[ -f "$OPD_ONLINE_RUN/queue/student/model.json" && -f "$OPD_ONLINE_RUN/queue/teacher/model.json" ]]; then
    break
  fi
  sleep 5
done
"$OPD_PYTHON" "$OPD_CODE_ROOT/ocrd_pipeline/online_rollout/run.py" \
  --input "${OPD_INPUT:-$OPD_ONLINE_RUN/prompts1000.jsonl}" \
  --output-dir "$OPD_ONLINE_RUN/results" --queue "$OPD_ONLINE_RUN/queue" \
  --tokenizer /storage/yukaichengLab/share/model/Qwen3-1.7B \
  --judge-config "$OPD_JUDGE_CONFIG" --secrets "$OPD_JUDGE_SECRETS" \
  --expected-rows "${OPD_EXPECTED_ROWS:-1000}" \
  --max-tokens "${OPD_MAX_TOKENS:-40264}" --concurrency "${OPD_CONCURRENCY:-8}"
touch "$OPD_ONLINE_RUN/GENERATION_COMPLETE"
