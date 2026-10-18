#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 MODE OUTPUT_ROOT" >&2
  exit 2
fi

MODE=$1
OUTPUT_ROOT=$2
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
WORKER=${ROOT}/ocrd_pipeline/full_rollout/run_full_rollout_worker.sh
LOG_ROOT=${OUTPUT_ROOT}/logs/workers_${SLURM_JOB_ID}
STATUS_ROOT=${LOG_ROOT}/status

mkdir -p "${LOG_ROOT}" "${STATUS_ROOT}"
trap 'rc=$?; echo "FULL_ROLLOUT_NODE8_ERROR mode=${MODE} job=${SLURM_JOB_ID:-unset} line=${LINENO} command=${BASH_COMMAND} exit_code=${rc}" >&2' ERR
echo "FULL_ROLLOUT_NODE8_START mode=${MODE} job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} time=$(date --iso-8601=seconds)"
/soft/slurm/bin/scontrol show job "${SLURM_JOB_ID}"
nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total --format=csv,noheader

gpu_count=$(nvidia-smi -L | wc -l)
if [[ "${gpu_count}" -ne 8 ]]; then
  echo "Expected exactly 8 visible GPUs, got ${gpu_count}" >&2
  exit 4
fi
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -v 'NVIDIA H800' >/dev/null; then
  echo "Allocated node contains a non-H800 GPU" >&2
  exit 5
fi
max_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | \
  awk 'BEGIN { max = 0 } { if ($1 > max) max = $1 } END { print max }')
if [[ "${max_used_mib}" -ge 500 ]]; then
  echo "GPU cleanliness invariant failed: max memory.used=${max_used_mib} MiB (limit <500 MiB)" >&2
  nvidia-smi
  exit 9
fi
other_jobs=$(/soft/slurm/bin/squeue -h -w "${SLURMD_NODENAME}" -o '%A' | sort -u | grep -vx "${SLURM_JOB_ID}" || true)
if [[ -n "${other_jobs}" ]]; then
  echo "Exclusive-node invariant failed: ${other_jobs}" >&2
  exit 6
fi
if [[ -f "${OUTPUT_ROOT}/FULL_ROLLOUT_BATCH_COMPLETE" ]]; then
  echo "Refusing to restart terminal output: ${OUTPUT_ROOT}" >&2
  exit 7
fi
echo "FULL_ROLLOUT_SINGLE_NODE_EXCLUSIVE_OK mode=${MODE} node=${SLURMD_NODENAME} gpu_count=${gpu_count} max_used_mib=${max_used_mib}"

PROBE=${OUTPUT_ROOT}/.write_probe_${SLURM_JOB_ID}
trap 'kill "${MONITOR_PID:-}" 2>/dev/null || true; unlink "${PROBE}" 2>/dev/null || true' EXIT
dd if=/dev/zero of="${PROBE}" bs=1M count=16 status=none
unlink "${PROBE}"
nvidia-smi dmon -d 10 -s pucm >"${LOG_ROOT}/gpu_dmon.log" 2>&1 &
MONITOR_PID=$!

run_worker_with_status() {
  local shard_id=$1
  local started_at finished_at rc temporary
  started_at=$(date --iso-8601=seconds)
  temporary=${STATUS_ROOT}/shard_${shard_id}.exit.tmp.$$
  trap 'rc=143; finished_at=$(date --iso-8601=seconds); printf "shard\t%s\nexit_code\t%s\nstarted_at\t%s\nfinished_at\t%s\nsignal\tTERM\n" "${shard_id}" "${rc}" "${started_at}" "${finished_at}" >"${temporary}"; mv "${temporary}" "${STATUS_ROOT}/shard_${shard_id}.exit"; exit "${rc}"' TERM
  set +e
  export CUDA_VISIBLE_DEVICES=${shard_id}
  bash "${WORKER}" "${MODE}" "${shard_id}" 8 "${OUTPUT_ROOT}"
  rc=$?
  set -e
  finished_at=$(date --iso-8601=seconds)
  printf 'shard\t%s\nexit_code\t%s\nstarted_at\t%s\nfinished_at\t%s\nsignal\tnone\n' \
    "${shard_id}" "${rc}" "${started_at}" "${finished_at}" >"${temporary}"
  mv "${temporary}" "${STATUS_ROOT}/shard_${shard_id}.exit"
  return "${rc}"
}

pids=()
for shard_id in $(seq 0 7); do
  run_worker_with_status "${shard_id}" >"${LOG_ROOT}/shard_${shard_id}.out" \
    2>"${LOG_ROOT}/shard_${shard_id}.err" &
  pids+=("$!")
done

failed=0
printf 'shard\tpid\texit_code\tstatus_file\n' >"${LOG_ROOT}/worker_exit_status.tsv"
set +e
for shard_id in $(seq 0 7); do
  wait "${pids[$shard_id]}"
  rc=$?
  printf '%s\t%s\t%s\t%s\n' "${shard_id}" "${pids[$shard_id]}" "${rc}" \
    "${STATUS_ROOT}/shard_${shard_id}.exit" >>"${LOG_ROOT}/worker_exit_status.tsv"
  if [[ "${rc}" -ne 0 ]]; then
    echo "FULL_ROLLOUT_WORKER_FAILED mode=${MODE} job=${SLURM_JOB_ID} shard=${shard_id} pid=${pids[$shard_id]} exit_code=${rc} status_file=${STATUS_ROOT}/shard_${shard_id}.exit" >&2
    failed=1
  fi
done
set -e
if [[ "${failed}" -ne 0 ]]; then
  echo "FULL_ROLLOUT_NODE8_FAILED mode=${MODE} job=${SLURM_JOB_ID} status_table=${LOG_ROOT}/worker_exit_status.tsv" >&2
  exit 8
fi
for shard_id in $(seq 0 7); do
  shard_name=$(printf '%02d' "${shard_id}")
  test -f "${OUTPUT_ROOT}/shards/shard_${shard_name}/SHARD_COMPLETE"
done
touch "${OUTPUT_ROOT}/GPU_GENERATION_COMPLETE"
echo "FULL_ROLLOUT_NODE8_COMPLETE mode=${MODE} job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} time=$(date --iso-8601=seconds)"
