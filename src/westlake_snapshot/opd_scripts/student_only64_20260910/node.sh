#!/bin/bash
set -Eeuo pipefail
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
test "${#devices[@]}" -eq 8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
pids=()
for ((localrank=0;localrank<8;localrank++)); do
 rank=$((SLURM_PROCID*8+localrank))
 (
  trap 'rc=$?; echo "WORKER_EXIT rank=$rank rc=$rc time=$(date -Is)"' EXIT
  export CUDA_VISIBLE_DEVICES=${devices[$localrank]}
  export VLLM_CACHE_ROOT=/tmp/s64_${SLURM_JOB_ID}_${rank}
  export TRITON_CACHE_DIR=/tmp/triton_s64_${SLURM_JOB_ID}_${rank}
  export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_s64_${SLURM_JOB_ID}_${rank}
  export VLLM_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
  setsid --wait python "$SCRIPT" generate --out "$OUT" --rank "$rank" --shards 16
 ) >"$ROOT/opd_logs/s64.${SLURM_JOB_ID}.gpu${rank}.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
test "$failed" -eq 0
