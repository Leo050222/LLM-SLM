#!/bin/bash
#SBATCH --job-name=deepmath_merge
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/deepmath_merge.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/deepmath_merge.%j.err

set -euo pipefail

source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
RUN=${ROOT}/opd_eval_deepmath/full_seed20260809

python ${ROOT}/opd_scripts/merge_deepmath_eval_diagnostic.py \
  --diagnostic-root ${RUN}/diagnostic \
  --teacher-output ${RUN}/qwen3_14b \
  --output-dir ${RUN}/merged

echo "DEEPMATH_FORMAL_COMPLETE=$(date --iso-8601=seconds)"
