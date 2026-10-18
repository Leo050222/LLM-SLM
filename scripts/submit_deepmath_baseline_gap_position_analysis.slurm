#!/bin/bash
#SBATCH --job-name=dm_gap_pos
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_gap_pos.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_gap_pos.%j.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
RUN=${ROOT}/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_baseline_seed20260811
python ${ROOT}/opd_scripts/analyze_deepmath_baseline_gap_position.py \
  --run-root ${RUN} \
  --score-root ${RUN}/sampled_token_logprob_gap \
  --tokenizer /storage/yukaichengLab/share/model/Qwen3-1.7B \
  --output ${RUN}/sampled_token_logprob_gap/position_analysis.json
echo "DEEPMATH_BASELINE_GAP_POSITION_COMPLETE=$(date --iso-8601=seconds)"
