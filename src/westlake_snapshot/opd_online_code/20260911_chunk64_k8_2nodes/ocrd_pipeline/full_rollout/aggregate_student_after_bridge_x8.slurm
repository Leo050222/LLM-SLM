#!/usr/bin/env bash
#SBATCH --job-name=sbridge_merge8
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/student_after_bridge_qwen3_1_7b_h256_n8_x8_seed20260826/logs/%j.aggregate.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/student_after_bridge_qwen3_1_7b_h256_n8_x8_seed20260826/logs/%j.aggregate.err

set -euo pipefail
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
PARENT=${ROOT}/dataset/train/student_after_bridge_qwen3_1_7b_h256_n8_x8_seed20260826
ORIGINAL=${ROOT}/dataset/train/student_after_bridge_qwen3_1_7b_h256_n8_seed20260826
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
"${PYTHON}" "${ROOT}/ocrd_pipeline/full_rollout/aggregate_student_after_bridge_x8.py" \
  --original-root "${ORIGINAL}" \
  --replicate-parent "${PARENT}" \
  --output-root "${PARENT}/aggregate"
test -f "${PARENT}/aggregate/STUDENT_AFTER_BRIDGE_X8_AGGREGATE_COMPLETE"
echo "STUDENT_AFTER_BRIDGE_ADD7_CHAIN_COMPLETE job=${SLURM_JOB_ID}"
