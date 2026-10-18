#!/bin/bash
#SBATCH --job-name=dm_nt_gap_vis
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_nt_gap_vis.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_nt_gap_vis.%j.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
RUN=${ROOT}/opd_eval_deepmath/ge6_test1000_native_nonthinking_seed20260811/qwen3_1_7b
python ${ROOT}/opd_scripts/build_deepmath_baseline_gap_vis.py \
  --run-root ${RUN} \
  --score-root ${RUN}/sampled_token_logprob_gap \
  --template ${ROOT}/opd_scripts/deepmath_baseline_gap_template.html \
  --output-dir ${RUN}/sampled_token_gap_visualization_compact \
  --tokenizer /storage/yukaichengLab/share/model/Qwen3-1.7B \
  --student-label "Qwen3-1.7B non-thinking" \
  --teacher-label "Qwen3-14B frozen" \
  --page-title "DeepMath Non-thinking · Sampled-token Log-probability Gap" \
  --page-heading "DeepMath non-thinking · sampled-token log-probability gap"
echo "DEEPMATH_NONTHINKING_GAP_VIS_COMPLETE=$(date --iso-8601=seconds)"
