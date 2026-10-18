#!/bin/bash
#SBATCH --job-name=opd_dm25_nt_rkl_verify
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/opd_dm25_nt_rkl_verify.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/opd_dm25_nt_rkl_verify.%j.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

RUN=/storage/yukaichengLab/wangxin/test_eiloeo/opd_runs/qwen3_1_7b_from_14b_deepmath_ge6_train25pct_nonthinking_student_top16_rkl_8h800_single_node
CKPT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_checkpoints/qwen3_1_7b_from_14b_deepmath_ge6_train25pct_nonthinking_student_top16_rkl_8h800_single_node
python /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/verify_deepmath_top16_smoke.py \
  "${RUN}" "${CKPT}" --expected-steps 222 \
  --expected-global-batch 64 --expected-checkpoints 44,89,133,178,222 \
  --expected-student-world-size 4 --expected-max-prompt-tokens 699 \
  --expected-thinking false --verify-hf-load
