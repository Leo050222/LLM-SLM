#!/bin/bash
set -Eeuo pipefail
"$OPD_PYTHON" "$SCALE_CODE/select_gpus.py"
export CUDA_VISIBLE_DEVICES=$(cat "$OPD_ONLINE_RUN/selected_gpu_uuids.txt")
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
test "${#devices[@]}" -eq "$((GRID_REPLICAS*2))"
pids=()
cleanup(){ for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT
for ((rank=0;rank<GRID_REPLICAS*2;rank++)); do
 VLLM_PORT=$((20000+rank*512)) SLURM_PROCID=$rank CUDA_VISIBLE_DEVICES=${devices[$rank]} bash "$SCALE_CODE/worker_rank.sh" >"$OPD_ONLINE_RUN/logs/gpu${rank}.log" 2>&1 &
 pids+=("$!")
done
"$OPD_PYTHON" "$SCALE_CODE/smoke.py" >"$OPD_ONLINE_RUN/logs/smoke.log" 2>&1 &
smoke_pid=$!
while kill -0 "$smoke_pid" 2>/dev/null; do
 for pid in "${pids[@]}"; do kill -0 "$pid" || { kill "$smoke_pid"; exit 1; }; done
 sleep 2
done
wait "$smoke_pid"
"$OPD_PYTHON" "$SCALE_CODE/runtime_smoke.py" >"$OPD_ONLINE_RUN/logs/runtime_smoke.log" 2>&1
"$OPD_PYTHON" "$SCALE_CODE/dispatch.py" >"$OPD_ONLINE_RUN/logs/dispatch.log" 2>&1 &
pids+=("$!")
while [[ ! -f "$OPD_ONLINE_RUN/SEGMENT_DONE.json" ]]; do
 test ! -f "$OPD_ONLINE_RUN/FAILED"
 for pid in "${pids[@]}"; do kill -0 "$pid"; done
 sleep 5
done
for ((k=0;k<GRID_REPLICAS;k++)); do touch "$OPD_ONLINE_RUN/queue/replica_$k/STOP"; done
for pid in "${pids[@]}"; do wait "$pid"; done
