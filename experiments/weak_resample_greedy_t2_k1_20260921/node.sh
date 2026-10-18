#!/bin/bash
set -Eeuo pipefail
export REPLICA_START=0
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
test "${#devices[@]}" -eq "$((GRID_REPLICAS*2))"
bash "$SCALE_CODE/preflight.sh"
pids=()
cleanup(){ for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT
for ((rank=0;rank<GRID_REPLICAS*2;rank++)); do
 VLLM_PORT=$((20000+rank*512)) SLURM_PROCID=$rank CUDA_VISIBLE_DEVICES=${devices[$rank]} bash "$SCALE_CODE/worker_rank.sh" >"$OPD_ONLINE_RUN/logs/gpu${rank}.log" 2>&1 &
 pids+=("$!")
done
run_checked_smoke(){
 "$OPD_PYTHON" "$SCALE_CODE/$1.py" >"$OPD_ONLINE_RUN/logs/$1.log" 2>&1 &
 smoke_pid=$!
 while kill -0 "$smoke_pid" 2>/dev/null; do
  for pid in "${pids[@]}"; do
   if ! kill -0 "$pid" 2>/dev/null; then
    echo "WORKER_EXIT_DURING_SMOKE pid=$pid"
    kill "$smoke_pid" 2>/dev/null || true
    wait "$smoke_pid" 2>/dev/null || true
    return 1
   fi
  done
  sleep 2
 done
 wait "$smoke_pid"
}
run_checked_smoke smoke
run_checked_smoke runtime_smoke
"$OPD_PYTHON" "$SCALE_CODE/dispatch.py" >"$OPD_ONLINE_RUN/logs/dispatch.log" 2>&1 &
pids+=("$!")
while [[ ! -f "$OPD_ONLINE_RUN/SEGMENT_DONE.json" ]]; do
 test ! -f "$OPD_ONLINE_RUN/FAILED"
 for pid in "${pids[@]}"; do kill -0 "$pid"; done
 sleep 5
done
for ((k=0;k<GRID_REPLICAS;k++)); do touch "$OPD_ONLINE_RUN/queue/replica_$k/STOP"; done
for pid in "${pids[@]}"; do wait "$pid"; done
