#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 OUTPUT_ROOT BASE_SEED MAX_RECORDS_OR_ZERO [BRIDGE_TOKEN_LIMIT_OR_ZERO]" >&2
  exit 2
fi

OUTPUT_ROOT=$1
BASE_SEED=$2
MAX_RECORDS=$3
BRIDGE_TOKEN_LIMIT=${4:-0}
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
WORKER=${ROOT}/ocrd_pipeline/full_rollout/run_student_after_bridge_replicate_worker.sh
LOG_ROOT=${OUTPUT_ROOT}/logs/workers_${SLURM_JOB_ID}
STATUS_ROOT=${LOG_ROOT}/status

mkdir -p "${LOG_ROOT}" "${STATUS_ROOT}"
trap 'rc=$?; echo "STUDENT_BRIDGE_REPLICATE_NODE8_ERROR job=${SLURM_JOB_ID:-unset} seed='"${BASE_SEED}"' line=${LINENO} command=${BASH_COMMAND} exit_code=${rc}" >&2' ERR
echo "STUDENT_BRIDGE_REPLICATE_NODE8_START job=${SLURM_JOB_ID} seed=${BASE_SEED} node=${SLURMD_NODENAME} time=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total --format=csv,noheader

gpu_count=$(nvidia-smi -L | wc -l)
[[ "${gpu_count}" -eq 8 ]] || { echo "Expected exactly 8 visible GPUs, got ${gpu_count}" >&2; exit 4; }
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -v 'NVIDIA H800' >/dev/null; then
  echo "Allocated node contains a non-H800 GPU" >&2
  exit 5
fi
max_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN { max=0 } { if ($1>max) max=$1 } END { print max }')
[[ "${max_used_mib}" -lt 500 ]] || { echo "GPU cleanliness invariant failed: max memory.used=${max_used_mib} MiB" >&2; exit 9; }
other_jobs=$(/soft/slurm/bin/squeue -h -w "${SLURMD_NODENAME}" -o '%A' | sort -u | grep -vx "${SLURM_JOB_ID}" || true)
[[ -z "${other_jobs}" ]] || { echo "Exclusive-node invariant failed: ${other_jobs}" >&2; exit 6; }
[[ ! -e "${OUTPUT_ROOT}/GPU_GENERATION_COMPLETE" ]] || { echo "Refusing to overwrite completed replicate: ${OUTPUT_ROOT}" >&2; exit 7; }
echo "STUDENT_BRIDGE_REPLICATE_NODE_EXCLUSIVE_OK node=${SLURMD_NODENAME} gpu_count=${gpu_count} max_used_mib=${max_used_mib}"

PROBE=${OUTPUT_ROOT}/.write_probe_${SLURM_JOB_ID}
trap 'kill "${MONITOR_PID:-}" 2>/dev/null || true; unlink "${PROBE}" 2>/dev/null || true' EXIT
dd if=/dev/zero of="${PROBE}" bs=1M count=16 status=none
unlink "${PROBE}"
nvidia-smi dmon -d 10 -s pucm >"${LOG_ROOT}/gpu_dmon.log" 2>&1 &
MONITOR_PID=$!

run_worker_with_status() {
  local shard_id=$1 started_at finished_at rc temporary
  started_at=$(date --iso-8601=seconds)
  temporary=${STATUS_ROOT}/shard_${shard_id}.exit.tmp.$$
  set +e
  export CUDA_VISIBLE_DEVICES=${shard_id}
  bash "${WORKER}" "${shard_id}" 8 "${OUTPUT_ROOT}" "${BASE_SEED}" "${MAX_RECORDS}" "${BRIDGE_TOKEN_LIMIT}"
  rc=$?
  set -e
  finished_at=$(date --iso-8601=seconds)
  printf 'shard\t%s\nexit_code\t%s\nstarted_at\t%s\nfinished_at\t%s\n' \
    "${shard_id}" "${rc}" "${started_at}" "${finished_at}" >"${temporary}"
  mv "${temporary}" "${STATUS_ROOT}/shard_${shard_id}.exit"
  return "${rc}"
}

pids=()
for shard_id in $(seq 0 7); do
  run_worker_with_status "${shard_id}" >"${LOG_ROOT}/shard_${shard_id}.out" 2>"${LOG_ROOT}/shard_${shard_id}.err" &
  pids+=("$!")
done

failed=0
printf 'shard\tpid\texit_code\tstatus_file\n' >"${LOG_ROOT}/worker_exit_status.tsv"
set +e
for shard_id in $(seq 0 7); do
  wait "${pids[$shard_id]}"
  rc=$?
  printf '%s\t%s\t%s\t%s\n' "${shard_id}" "${pids[$shard_id]}" "${rc}" "${STATUS_ROOT}/shard_${shard_id}.exit" >>"${LOG_ROOT}/worker_exit_status.tsv"
  [[ "${rc}" -eq 0 ]] || failed=1
done
set -e
[[ "${failed}" -eq 0 ]] || { echo "STUDENT_BRIDGE_REPLICATE_NODE8_FAILED status=${LOG_ROOT}/worker_exit_status.tsv" >&2; exit 8; }
for shard_id in $(seq 0 7); do
  test -f "${OUTPUT_ROOT}/shards/shard_$(printf '%02d' "${shard_id}")/SHARD_COMPLETE"
done
touch "${OUTPUT_ROOT}/GPU_GENERATION_COMPLETE"
echo "STUDENT_BRIDGE_REPLICATE_NODE8_COMPLETE job=${SLURM_JOB_ID} seed=${BASE_SEED} node=${SLURMD_NODENAME} time=$(date --iso-8601=seconds)"
