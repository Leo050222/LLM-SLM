#!/usr/bin/env bash
#SBATCH --job-name=sbridge_horz_score
#SBATCH --partition=yukaichenglab
#SBATCH --array=0-23%24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/sbridge_horz_score.%A_%a.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/sbridge_horz_score.%A_%a.err

set -euo pipefail
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
HORIZONS=(32 64 128)
HORIZON_INDEX=$((SLURM_ARRAY_TASK_ID / 8))
REPLICATE=$((SLURM_ARRAY_TASK_ID % 8))
H=${HORIZONS[${HORIZON_INDEX}]}
PARENT=${ROOT}/dataset/train/student_after_bridge_qwen3_1_7b_h${H}_n8_x8_seed20260826
ROLLOUT=${PARENT}/$(printf 'replicate_%02d' "${REPLICATE}")
BASE_SEED=$((20260826 + REPLICATE * 1000000))

test -f "${PARENT}/GPU_ALL_REPLICATES_COMPLETE"
bash "${ROOT}/ocrd_pipeline/full_rollout/score_student_after_bridge_horizon_root.sh" \
  "${ROLLOUT}" "${BASE_SEED}" "${H}"
