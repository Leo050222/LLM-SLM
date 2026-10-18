#!/bin/bash
set -euo pipefail
: "${OPD_CODE_ROOT:?}"
: "${OPD_ONLINE_RUN:?}"
: "${OPD_PYTHON:?}"
: "${OPD_JUDGE_CONFIG:?}"
: "${OPD_JUDGE_SECRETS:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
stop_workers() {
 for replica in $(seq 0 3); do
  mkdir -p "$OPD_ONLINE_RUN/queue/replica_$replica"
  touch "$OPD_ONLINE_RUN/queue/replica_$replica/STOP"
 done
}
trap stop_workers EXIT
ready=0
for ((i=0; i<360; i++)); do
 ready=1
 for replica in $(seq 0 3); do
  for role in student teacher; do
   if [[ ! -f "$OPD_ONLINE_RUN/queue/replica_$replica/$role/model.json" ]]; then ready=0; fi
  done
 done
 if (( ready )); then break; fi
 sleep 5
done
if (( ! ready )); then echo 'Worker readiness timed out' >&2; exit 1; fi
for attempt in $(seq 1 6); do
 echo "COORDINATOR_PASS $attempt"
 set +e
 "$OPD_PYTHON" "$OPD_CODE_ROOT/ocrd_pipeline/online_rollout/run.py" \
 --input "$OPD_ONLINE_RUN/prompts1000.jsonl" \
 --output-dir "$OPD_ONLINE_RUN/results" --queue "$OPD_ONLINE_RUN/queue" --replicas 4 \
 --tokenizer /storage/yukaichengLab/share/model/Qwen3-1.7B \
 --judge-config "$OPD_JUDGE_CONFIG" --secrets "$OPD_JUDGE_SECRETS" \
 --expected-rows 1000 --max-tokens 40264 --concurrency 128 --teacher-tokens 32
rc=$?
 set -e
 if (( rc == 0 )); then touch "$OPD_ONLINE_RUN/GENERATION_COMPLETE"; exit 0; fi
 echo "COORDINATOR_RETRY exit=$rc"
 sleep 30
done
exit 1
