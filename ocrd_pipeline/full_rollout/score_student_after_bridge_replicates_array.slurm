#!/usr/bin/env bash
#SBATCH --job-name=sbridge_score7
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --array=1-7
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/student_after_bridge_qwen3_1_7b_h256_n8_x8_seed20260826/logs/%A_%a.score.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/student_after_bridge_qwen3_1_7b_h256_n8_x8_seed20260826/logs/%A_%a.score.err

set -euo pipefail
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
PARENT=${ROOT}/dataset/train/student_after_bridge_qwen3_1_7b_h256_n8_x8_seed20260826
replicate=${SLURM_ARRAY_TASK_ID}
ROLLOUT=${PARENT}/$(printf 'replicate_%02d' "${replicate}")
BASE_SEED=$((20260826 + replicate * 1000000))
test -f "${PARENT}/GPU_ALL_REPLICATES_COMPLETE"
bash "${ROOT}/ocrd_pipeline/full_rollout/score_student_after_bridge_root.sh" "${ROLLOUT}" "${BASE_SEED}"
