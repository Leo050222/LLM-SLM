#!/bin/bash
set -Eeuo pipefail
export GRID_SHARD=${SLURM_NODEID:?}
export OPD_ONLINE_RUN=$GRID_ROOT/slots/job_${SLURM_JOB_ID}_node_${GRID_SHARD}
mkdir -p "$OPD_ONLINE_RUN/logs"
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
test "${#devices[@]}" -eq 8
pids=()
cleanup(){ for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT
# Initialize teachers first, then colocated students, independently per GPU.
for role in teacher student; do
 for ((gpu=0;gpu<8;gpu++)); do
  rank=$((gpu*2))
  [[ "$role" != teacher ]] || rank=$((rank+1))
  VLLM_PORT=$((20000+rank*512)) SLURM_PROCID=$rank CUDA_VISIBLE_DEVICES=${devices[$gpu]} bash "$SCALE_CODE/worker_rank.sh" >"$OPD_ONLINE_RUN/logs/gpu${gpu}_${role}.log" 2>&1 &
  pids+=("$!")
 done
 deadline=$((SECONDS+1800))
 while true; do
  ready=0
  for ((gpu=0;gpu<8;gpu++)); do
   [[ ! -f "$OPD_ONLINE_RUN/queue/replica_$gpu/$role/ready.json" ]] || ready=$((ready+1))
  done
  (( ready != 8 )) || break
  for pid in "${pids[@]}"; do kill -0 "$pid"; done
  (( SECONDS < deadline )) || { echo "MODEL_STARTUP_TIMEOUT $role"; exit 1; }
  sleep 5
 done
done
"$OPD_PYTHON" "$SCALE_CODE/smoke.py" >"$OPD_ONLINE_RUN/logs/smoke.log" 2>&1
# Retain the same GPU workers and allocation across dispatcher passes.
coordinate_until_complete(){
 while [[ ! -f "$OPD_ONLINE_RUN/SHARD_COMPLETE.json" ]]; do
  rm -f "$OPD_ONLINE_RUN/SEGMENT_DONE.json"
  "$OPD_PYTHON" "$SCALE_CODE/dispatch.py" >>"$OPD_ONLINE_RUN/logs/dispatch.log" 2>&1
  test -f "$OPD_ONLINE_RUN/SEGMENT_DONE.json"
 done
 touch "$OPD_ONLINE_RUN/ALLOCATION_DONE"
}
coordinate_until_complete &
pids+=("$!")
while [[ ! -f "$OPD_ONLINE_RUN/ALLOCATION_DONE" ]]; do
 test ! -f "$OPD_ONLINE_RUN/FAILED"
 for pid in "${pids[@]}"; do kill -0 "$pid"; done
 sleep 5
done
for ((k=0;k<GRID_REPLICAS;k++)); do touch "$OPD_ONLINE_RUN/queue/replica_$k/STOP"; done
for pid in "${pids[@]}"; do wait "$pid"; done
