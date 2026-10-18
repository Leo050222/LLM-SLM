#!/bin/bash
#SBATCH --job-name=dm_s16_merge
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_s16_merge.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_s16_merge.%j.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
RUN=${ROOT}/opd_eval_deepmath/maxlen_seed20260809
python ${ROOT}/opd_scripts/merge_deepmath_student_top16_trace_diagnostic.py \
  --root ${RUN}/student_top16_trace_diagnostic \
  --output-dir ${RUN}/student_top16_trace_diagnostic_merged
python ${ROOT}/opd_scripts/build_deepmath_student_top16_trace_vis.py \
  --run-root ${RUN} \
  --diagnostic-root ${RUN}/student_top16_trace_diagnostic_merged \
  --template ${ROOT}/opd_scripts/student_top16_trace_template.html \
  --output ${RUN}/deepmath_student_top16_trace_difference.html
echo "STUDENT_TOP16_TRACE_MERGE_COMPLETE=$(date --iso-8601=seconds)"
