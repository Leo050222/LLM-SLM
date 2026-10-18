#!/bin/bash
#SBATCH --job-name=dm17_t16_vis
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm17_t16_vis.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm17_t16_vis.%j.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
RUN=${ROOT}/opd_eval_deepmath/maxlen_seed20260809

python ${ROOT}/opd_scripts/merge_deepmath_eval_diagnostic.py \
  --diagnostic-root ${RUN}/diagnostic_qwen3_1_7b_top16 \
  --teacher-output ${RUN}/qwen3_14b \
  --student-key qwen3_1_7b \
  --expected-student /storage/yukaichengLab/share/model/Qwen3-1.7B \
  --expected-max-tokens 40512 --expected-max-model-len 40960 --expected-topk 16 \
  --output-dir ${RUN}/merged_qwen3_1_7b_top16

python ${ROOT}/opd_scripts/build_deepmath_17b_miss14b_token_vis.py \
  --root ${RUN} \
  --diagnostic-dir diagnostic_qwen3_1_7b_top16 \
  --merged-dir merged_qwen3_1_7b_top16 \
  --support-topk 16 \
  --template ${ROOT}/opd_scripts/deepmath_17b_miss14b_fragment.html \
  --output ${RUN}/deepmath_17b_miss14b_token_difference_top16.html

echo "DEEPMATH_17B_TOP16_VIS_COMPLETE=$(date --iso-8601=seconds)"
