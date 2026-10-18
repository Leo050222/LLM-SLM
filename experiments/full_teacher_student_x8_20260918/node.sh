#!/bin/bash
set -Eeuo pipefail
node_rank=${SLURM_NODEID:?}
[[ "$node_rank" == 0 ]]
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
test "${#devices[@]}" -eq 4
printf 'NODE_BINDING node_rank=%s host=%s devices=%s\n' "$node_rank" "$SLURMD_NODENAME" "$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,uuid,name,memory.used --format=csv > "$OPD_ONLINE_RUN/logs/node${node_rank}_gpus.csv"
pids=()
cleanup(){ for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT
for ((local_rank=0;local_rank<4;local_rank++)); do
 rank=$((node_rank*4+local_rank))
 VLLM_PORT=$((20000+rank*512)) SLURM_PROCID=$rank CUDA_VISIBLE_DEVICES=${devices[$local_rank]} bash "$SCALE_CODE/worker_rank.sh" >"$OPD_ONLINE_RUN/logs/gpu${rank}.log" 2>&1 &
 pids+=("$!")
done
if (( node_rank == 0 )); then
 coordinator(){
  "$OPD_PYTHON" "$SCALE_CODE/smoke.py" >"$OPD_ONLINE_RUN/logs/smoke.log" 2>&1
  "$OPD_PYTHON" "$SCALE_CODE/dispatch.py" >"$OPD_ONLINE_RUN/logs/dispatch.log" 2>&1
 }
 coordinator &
 pids+=("$!")
fi
while [[ ! -f "$OPD_ONLINE_RUN/SEGMENT_DONE.json" ]]; do
 test ! -f "$OPD_ONLINE_RUN/FAILED"
 for pid in "${pids[@]}"; do kill -0 "$pid"; done
 sleep 5
done
for ((k=node_rank*4;k<node_rank*4+4;k++)); do touch "$OPD_ONLINE_RUN/queue/replica_$k/STOP"; done
for pid in "${pids[@]}"; do wait "$pid"; done
