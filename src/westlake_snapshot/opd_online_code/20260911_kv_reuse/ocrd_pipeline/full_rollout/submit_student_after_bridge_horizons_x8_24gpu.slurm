#!/usr/bin/env bash
#SBATCH --job-name=sbridge_horz_x8
#SBATCH --partition=yukaichenglab
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --gres=gpu:H800:8
#SBATCH --exclusive
#SBATCH --exclude=gnho006,gnho009,gnho017,gnho018,gnho032,gnho036,gnho038
#SBATCH --time=3-00:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/sbridge_horz_x8.%A_%a.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/sbridge_horz_x8.%A_%a.err

set -euo pipefail
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
HORIZONS=(32 64 128)
H=${HORIZONS[${SLURM_ARRAY_TASK_ID}]}
PARENT=${ROOT}/dataset/train/student_after_bridge_qwen3_1_7b_h${H}_n8_x8_seed20260826

[[ ! -e "${PARENT}/GPU_ALL_REPLICATES_COMPLETE" ]] || {
  echo "Refusing to overwrite terminal parent ${PARENT}" >&2
  exit 7
}
mkdir -p "${PARENT}/logs"
echo "STUDENT_BRIDGE_HORIZON_X8_START horizon=${H} job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} time=$(date --iso-8601=seconds)"
bash "${ROOT}/ocrd_pipeline/full_rollout/run_student_after_bridge_replicate_group.sh" \
  "${PARENT}" 0 7 0 "${H}"
for replicate in $(seq 0 7); do
  test -f "${PARENT}/$(printf 'replicate_%02d' "${replicate}")/GPU_GENERATION_COMPLETE"
done
touch "${PARENT}/GPU_ALL_REPLICATES_COMPLETE"
echo "STUDENT_BRIDGE_HORIZON_X8_COMPLETE horizon=${H} job=${SLURM_JOB_ID} time=$(date --iso-8601=seconds)"
