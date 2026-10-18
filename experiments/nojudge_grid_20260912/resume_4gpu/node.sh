#!/bin/bash
set -Eeuo pipefail
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
test "${#devices[@]}" -eq 4
pids=()
cleanup(){ for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT
node_rank=$SLURM_PROCID
for ((j=0;j<4;j++)); do
 rank=$((node_rank*4+j))
 VLLM_PORT=$((20000+rank*512)) SLURM_PROCID=$rank CUDA_VISIBLE_DEVICES=${devices[$j]} bash "$OPD_CODE_ROOT/ocrd_pipeline/online_rollout/worker_rank.sh" >"$OPD_ONLINE_RUN/logs/gpu${rank}.log" 2>&1 &
 pids+=("$!")
done
"$OPD_PYTHON" /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/nojudge_grid_4gpu_20260912/dispatch.py >"$OPD_ONLINE_RUN/logs/dispatch.log" 2>&1 &
pids+=("$!")
while [[ ! -f "$OPD_ONLINE_RUN/ALL_GENERATION_COMPLETE.json" ]]; do
 test ! -f "$OPD_ONLINE_RUN/FAILED"
 for pid in "${pids[@]}"; do kill -0 "$pid"; done
 sleep 5
done
for ((k=node_rank*2;k<(node_rank+1)*2;k++)); do touch "$OPD_ONLINE_RUN/queue/replica_$k/STOP"; done
for pid in "${pids[@]}"; do wait "$pid"; done
