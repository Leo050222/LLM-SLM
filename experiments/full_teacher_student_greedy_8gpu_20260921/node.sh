#!/bin/bash
set -Eeuo pipefail
node_rank=${SLURM_NODEID:?}
[[ "$node_rank" == 0 ]]
test "$SLURMD_NODENAME" != gnho008
nvidia-smi --query-gpu=index,uuid,name,memory.used --format=csv > "$OPD_ONLINE_RUN/logs/node${node_rank}_gpus.csv"
"$OPD_PYTHON" "$SCALE_CODE/select_gpus.py" > "$OPD_ONLINE_RUN/logs/selection.log" 2>&1
mapfile -t devices < "$OPD_ONLINE_RUN/healthy_gpu_uuids.txt"
mapfile -t cuda_indices < "$OPD_ONLINE_RUN/healthy_gpu_indices.txt"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export GRID_REPLICAS=${#devices[@]}
test "$GRID_REPLICAS" -eq 8
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${devices[*]}")
printf 'NODE_BINDING host=%s replicas=%s uuid=%s\n' "$SLURMD_NODENAME" "$GRID_REPLICAS" "$CUDA_VISIBLE_DEVICES"
bash "$SCALE_CODE/preflight.sh"
if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then exit 0; fi
pids=()
cleanup(){ for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT
for ((local_rank=0;local_rank<GRID_REPLICAS;local_rank++)); do
 rank=$((local_rank))
 VLLM_PORT=$((20000+rank*512)) SLURM_PROCID=$rank CUDA_VISIBLE_DEVICES=${cuda_indices[$local_rank]} bash "$SCALE_CODE/worker_rank.sh" >"$OPD_ONLINE_RUN/logs/gpu${rank}.log" 2>&1 &
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
for ((k=0;k<GRID_REPLICAS;k++)); do touch "$OPD_ONLINE_RUN/queue/replica_$k/STOP"; done
for pid in "${pids[@]}"; do wait "$pid"; done
