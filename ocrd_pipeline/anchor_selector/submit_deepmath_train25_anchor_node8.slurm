#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H800:8
#SBATCH --exclude=gnho009,gnho018
#SBATCH --cpus-per-task=112
#SBATCH --mem=1200G
#SBATCH --time=08:00:00
#SBATCH -J anchor14k_n8
#SBATCH -o /storage/yukaichengLab/wangxin/test_eiloeo/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/logs/%j_node8.out
#SBATCH -e /storage/yukaichengLab/wangxin/test_eiloeo/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/logs/%j_node8.err

set -euo pipefail

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
MODULE=${ROOT}/ocrd_pipeline/anchor_selector
RUN_ROOT=${ROOT}/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2
WORKER=${MODULE}/run_deepmath_train25_anchor_shard_worker.sh

if [[ ! -f "${RUN_ROOT}/shards/shard_00/SHARD_COMPLETE" ]]; then
  echo "Missing successful shard-0 benchmark output" >&2
  exit 1
fi

echo "ANCHOR14K_NODE8_START job=${SLURM_JOB_ID} node=${SLURMD_NODENAME}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

run_wave() {
  local tasks=("$@")
  local pids=()
  local task
  for task in "${tasks[@]}"; do
    srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:H800:1 \
      --cpus-per-task=14 --mem=140G \
      bash "${WORKER}" "${task}" \
      >"${RUN_ROOT}/logs/${SLURM_JOB_ID}_${task}.out" \
      2>"${RUN_ROOT}/logs/${SLURM_JOB_ID}_${task}.err" &
    pids+=("$!")
  done
  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "ANCHOR14K_NODE8_WAVE_FAILED tasks=${tasks[*]}" >&2
    exit 1
  fi
  echo "ANCHOR14K_NODE8_WAVE_COMPLETE tasks=${tasks[*]}"
}

run_wave 1 2 3 4 5 6 7 8
run_wave 9 10 11 12 13 14 15

for task in $(seq -w 0 15); do
  test -f "${RUN_ROOT}/shards/shard_${task}/SHARD_COMPLETE"
done
touch "${RUN_ROOT}/GPU_SCORING_COMPLETE"
echo "ANCHOR14K_NODE8_COMPLETE job=${SLURM_JOB_ID} node=${SLURMD_NODENAME}"
