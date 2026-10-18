#!/usr/bin/env bash
#SBATCH --job-name=sbridge_add7_2n
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --gres=gpu:H800:8
#SBATCH --exclusive
#SBATCH --exclude=gnho006,gnho009,gnho017,gnho018,gnho036
#SBATCH --time=16:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/student_after_bridge_qwen3_1_7b_h256_n8_x8_seed20260826/logs/%j.gpu.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/student_after_bridge_qwen3_1_7b_h256_n8_x8_seed20260826/logs/%j.gpu.err

set -euo pipefail
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
PARENT=${ROOT}/dataset/train/student_after_bridge_qwen3_1_7b_h256_n8_x8_seed20260826
GROUP=${ROOT}/ocrd_pipeline/full_rollout/run_student_after_bridge_replicate_group.sh
mkdir -p "${PARENT}/logs"
[[ ! -e "${PARENT}/GPU_ALL_REPLICATES_COMPLETE" ]] || { echo "Refusing to overwrite terminal parent ${PARENT}" >&2; exit 7; }
mapfile -t NODES < <(/soft/slurm/bin/scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ "${#NODES[@]}" -eq 2 ]] || { echo "Expected two nodes, got ${#NODES[@]}" >&2; exit 4; }
echo "STUDENT_BRIDGE_ADD7_2NODE_START job=${SLURM_JOB_ID} nodes=${NODES[*]} time=$(date --iso-8601=seconds)"

/soft/slurm/bin/srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=64 --gpus-per-node=8 \
  --nodelist="${NODES[0]}" bash "${GROUP}" "${PARENT}" 1 4 0 \
  >"${PARENT}/logs/${SLURM_JOB_ID}.node0.out" 2>"${PARENT}/logs/${SLURM_JOB_ID}.node0.err" &
pid0=$!
/soft/slurm/bin/srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=64 --gpus-per-node=8 \
  --nodelist="${NODES[1]}" bash "${GROUP}" "${PARENT}" 5 7 0 \
  >"${PARENT}/logs/${SLURM_JOB_ID}.node1.out" 2>"${PARENT}/logs/${SLURM_JOB_ID}.node1.err" &
pid1=$!

set +e
wait "${pid0}"; rc0=$?
wait "${pid1}"; rc1=$?
set -e
printf 'node\treplicates\texit_code\n%s\t1-4\t%s\n%s\t5-7\t%s\n' \
  "${NODES[0]}" "${rc0}" "${NODES[1]}" "${rc1}" >"${PARENT}/logs/${SLURM_JOB_ID}.node_status.tsv"
[[ "${rc0}" -eq 0 && "${rc1}" -eq 0 ]] || { echo "Two-node generation failed: rc0=${rc0} rc1=${rc1}" >&2; exit 8; }
for replicate in $(seq 1 7); do
  test -f "${PARENT}/$(printf 'replicate_%02d' "${replicate}")/GPU_GENERATION_COMPLETE"
done
touch "${PARENT}/GPU_ALL_REPLICATES_COMPLETE"
echo "STUDENT_BRIDGE_ADD7_2NODE_COMPLETE job=${SLURM_JOB_ID} time=$(date --iso-8601=seconds)"
