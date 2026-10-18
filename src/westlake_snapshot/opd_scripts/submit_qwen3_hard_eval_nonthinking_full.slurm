#!/bin/bash
#SBATCH --job-name=qwen_hard_nt_full
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:H800:1
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/qwen_hard_nt_full.%A_%a.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/qwen_hard_nt_full.%A_%a.err

set -euo pipefail

case "${SLURM_ARRAY_TASK_ID}" in
  0) MODEL_TAG=qwen3_4b ;;
  1) MODEL_TAG=qwen3_14b ;;
  *) echo "Unexpected array task: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

bash /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/run_qwen3_hard_eval_nonthinking.sh "${MODEL_TAG}"
