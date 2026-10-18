#!/usr/bin/env bash
#SBATCH --job-name=sbridge_horz_merge
#SBATCH --partition=yukaichenglab
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/sbridge_horz_merge.%A_%a.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/sbridge_horz_merge.%A_%a.err

set -euo pipefail
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
HORIZONS=(32 64 128)
H=${HORIZONS[${SLURM_ARRAY_TASK_ID}]}
PARENT=${ROOT}/dataset/train/student_after_bridge_qwen3_1_7b_h${H}_n8_x8_seed20260826

"${PYTHON}" "${ROOT}/ocrd_pipeline/full_rollout/aggregate_student_after_bridge_x8.py" \
  --original-root "${PARENT}/replicate_00" \
  --replicate-parent "${PARENT}" \
  --output-root "${PARENT}/aggregate" \
  --bridge-token-limit "${H}"
test -f "${PARENT}/aggregate/STUDENT_AFTER_BRIDGE_X8_AGGREGATE_COMPLETE"
echo "STUDENT_BRIDGE_HORIZON_X8_AGGREGATE_COMPLETE horizon=${H} job=${SLURM_JOB_ID}"
