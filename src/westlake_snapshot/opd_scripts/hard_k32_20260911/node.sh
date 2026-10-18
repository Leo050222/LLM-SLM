#!/bin/bash
set -Eeuo pipefail
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
test "${#devices[@]}" -eq 8
pids=()
cleanup(){ for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT
node_rank=$SLURM_PROCID
for ((j=0;j<8;j++)); do
 rank=$((node_rank*8+j))
 SLURM_PROCID=$rank CUDA_VISIBLE_DEVICES=${devices[$j]} bash "$OPD_CODE_ROOT/ocrd_pipeline/online_rollout/worker_rank.sh" >"$OPD_ONLINE_RUN/logs/gpu${rank}.log" 2>&1 &
 pids+=("$!")
done
while [[ ! -f "$OPD_ONLINE_RUN/ALL_GENERATION_COMPLETE.json" ]]; do
 test ! -f "$OPD_ONLINE_RUN/FAILED"
 for pid in "${pids[@]}"; do kill -0 "$pid"; done
 sleep 5
done
for ((k=node_rank*4;k<(node_rank+1)*4;k++)); do touch "$OPD_ONLINE_RUN/queue/replica_$k/STOP"; done
for pid in "${pids[@]}"; do wait "$pid"; done
