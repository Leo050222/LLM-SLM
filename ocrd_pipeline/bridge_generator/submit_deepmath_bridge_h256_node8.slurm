#!/usr/bin/env bash
#SBATCH --job-name=bridge256_n8
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --gres=gpu:H800:8
#SBATCH --exclusive
#SBATCH --exclude=gnho009,gnho018,gnho036
#SBATCH --time=08:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/bridge_qwen3_14b_h256_n8_seed20260822/logs/%j.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/bridge_qwen3_14b_h256_n8_seed20260822/logs/%j.err

set -euo pipefail

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
MODULE=${ROOT}/ocrd_pipeline/bridge_generator
OUTPUT_ROOT=${ROOT}/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/bridge_qwen3_14b_h256_n8_seed20260822
WORKER=${MODULE}/run_bridge_h256_shard_worker.sh
LOG_ROOT=${OUTPUT_ROOT}/logs/workers_${SLURM_JOB_ID}

mkdir -p "${LOG_ROOT}"
echo "BRIDGE_NODE8_START job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} time=$(date --iso-8601=seconds)"
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
other_jobs=$(/soft/slurm/bin/squeue -h -w "${SLURMD_NODENAME}" -o '%A' | sort -u | grep -vx "${SLURM_JOB_ID}" || true)
if [[ -n "${other_jobs}" ]]; then
  echo "Exclusive-node invariant failed: ${other_jobs}" >&2
  exit 6
fi
echo "BRIDGE_SINGLE_NODE_EXCLUSIVE_OK node=${SLURMD_NODENAME} gpu_count=${gpu_count}"

PROBE=${OUTPUT_ROOT}/.write_probe_${SLURM_JOB_ID}
trap 'kill "${MONITOR_PID:-}" 2>/dev/null || true; unlink "${PROBE}" 2>/dev/null || true' EXIT
dd if=/dev/zero of="${PROBE}" bs=1M count=16 status=none
unlink "${PROBE}"
nvidia-smi dmon -d 10 -s pucm >"${LOG_ROOT}/gpu_dmon.log" 2>&1 &
MONITOR_PID=$!

pids=()
for shard_id in $(seq 0 7); do
  (
    export CUDA_VISIBLE_DEVICES=${shard_id}
    bash "${WORKER}" "${shard_id}" 8
  ) >"${LOG_ROOT}/shard_${shard_id}.out" \
    2>"${LOG_ROOT}/shard_${shard_id}.err" &
  pids+=("$!")
done

failed=0
for shard_id in $(seq 0 7); do
  if ! wait "${pids[$shard_id]}"; then
    echo "BRIDGE_WORKER_FAILED shard=${shard_id}" >&2
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  exit 8
fi
for shard_id in $(seq 0 7); do
  shard_name=$(printf '%02d' "${shard_id}")
  test -f "${OUTPUT_ROOT}/shards/shard_${shard_name}/SHARD_COMPLETE"
done
touch "${OUTPUT_ROOT}/GPU_GENERATION_COMPLETE"
echo "BRIDGE_NODE8_COMPLETE job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} time=$(date --iso-8601=seconds)"
