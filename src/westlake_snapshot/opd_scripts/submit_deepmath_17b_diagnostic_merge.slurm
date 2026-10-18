#!/bin/bash
#SBATCH --job-name=deepmath17_merge
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/deepmath17_merge.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/deepmath17_merge.%j.err

set -euo pipefail

source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
RUN=${ROOT}/opd_eval_deepmath/full_seed20260809

python ${ROOT}/opd_scripts/merge_deepmath_eval_diagnostic.py \
  --diagnostic-root ${RUN}/diagnostic_qwen3_1_7b \
  --teacher-output ${RUN}/qwen3_14b \
  --student-key qwen3_1_7b \
  --expected-student /storage/yukaichengLab/share/model/Qwen3-1.7B \
  --output-dir ${RUN}/merged_qwen3_1_7b

echo "DEEPMATH_17B_FORMAL_COMPLETE=$(date --iso-8601=seconds)"
