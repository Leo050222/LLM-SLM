#!/usr/bin/env bash
#SBATCH --job-name=verify_goodbr_top16
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/verify_good_bridge_top16.%j.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/verify_good_bridge_top16.%j.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
SELECTION=${ROOT}/dataset/train/student_good_bridge_selection_qwen3_1_7b_h256_n8_seed20260827/selection.json
python ${ROOT}/opd_scripts/verify_bridge_top16_reverse_kl_train.py \
  ${ROOT}/opd_runs/qwen3_1_7b_bridge_h256_good37724_student_top16_rkl_seed20260827 \
  ${ROOT}/opd_checkpoints/qwen3_1_7b_bridge_h256_good37724_student_top16_rkl_seed20260827 \
  --expected-steps 590 --expected-checkpoints 118,236,354,472,590 \
  --expected-covered-bridges 37724 --expected-train-rows 37724 \
  --expected-selection "${SELECTION}" --require-selected-epoch --verify-hf-load
echo GOOD_BRIDGE_TOP16_RKL_FORMAL_VERIFY_COMPLETE
