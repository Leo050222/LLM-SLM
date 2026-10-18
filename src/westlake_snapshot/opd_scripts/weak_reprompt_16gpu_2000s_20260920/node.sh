#!/bin/bash
set -Eeuo pipefail
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
test "${#devices[@]}" -eq 8
bash "$SCALE_CODE/preflight.sh"
pids=()
cleanup(){ for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT
for ((rank=0;rank<8;rank++)); do
 VLLM_PORT=$((20000+rank*512)) SLURM_PROCID=$((SLURM_NODEID*8+rank)) CUDA_VISIBLE_DEVICES=${devices[$rank]} bash "$SCALE_CODE/worker_rank.sh" >"$OPD_ONLINE_RUN/logs/node${SLURM_NODEID}_gpu${rank}.log" 2>&1 &
 pids+=("$!")
done
export REPLICA_START=$((SLURM_NODEID*4))
"$OPD_PYTHON" "$SCALE_CODE/smoke.py" >"$OPD_ONLINE_RUN/logs/node${SLURM_NODEID}_smoke.log" 2>&1
"$OPD_PYTHON" "$SCALE_CODE/runtime_smoke.py" >"$OPD_ONLINE_RUN/logs/node${SLURM_NODEID}_runtime_smoke.log" 2>&1
touch "$OPD_ONLINE_RUN/node${SLURM_NODEID}_SMOKE_OK"
if [[ "$SLURM_NODEID" == 0 ]]; then
 for ((attempt=0;attempt<720;attempt++)); do
  [[ ! -f "$OPD_ONLINE_RUN/node1_SMOKE_OK" ]] || break
  for pid in "${pids[@]}"; do kill -0 "$pid"; done
  sleep 5
 done
 test -f "$OPD_ONLINE_RUN/node1_SMOKE_OK"
 "$OPD_PYTHON" "$SCALE_CODE/dispatch.py" >"$OPD_ONLINE_RUN/logs/dispatch.log" 2>&1 &
 pids+=("$!")
fi
while [[ ! -f "$OPD_ONLINE_RUN/SEGMENT_DONE.json" ]]; do
 test ! -f "$OPD_ONLINE_RUN/FAILED"
 for pid in "${pids[@]}"; do kill -0 "$pid"; done
 sleep 5
done
for ((k=REPLICA_START;k<REPLICA_START+4;k++)); do touch "$OPD_ONLINE_RUN/queue/replica_$k/STOP"; done
for pid in "${pids[@]}"; do wait "$pid"; done
