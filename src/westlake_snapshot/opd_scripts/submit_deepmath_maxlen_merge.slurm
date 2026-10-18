#!/bin/bash
#SBATCH --job-name=dm_max_merge
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --array=0-1
#SBATCH --time=08:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_max_merge.%A_%a.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_max_merge.%A_%a.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
RUN=${ROOT}/opd_eval_deepmath/maxlen_seed20260809
TASK=${SLURM_ARRAY_TASK_ID}
if [[ "${TASK}" -eq 0 ]]; then
  DIAG=${RUN}/diagnostic
  KEY=qwen3_4b
  STUDENT=/storage/yukaichengLab/share/model/Qwen3-4B
  OUT=${RUN}/merged
else
  DIAG=${RUN}/diagnostic_qwen3_1_7b
  KEY=qwen3_1_7b
  STUDENT=/storage/yukaichengLab/share/model/Qwen3-1.7B
  OUT=${RUN}/merged_qwen3_1_7b
fi
python ${ROOT}/opd_scripts/merge_deepmath_eval_diagnostic.py \
  --diagnostic-root ${DIAG} --teacher-output ${RUN}/qwen3_14b \
  --student-key ${KEY} --expected-student ${STUDENT} --output-dir ${OUT} \
  --expected-max-tokens 40512 --expected-max-model-len 40960
echo "MAXLEN_MERGE_COMPLETE=${TASK} $(date --iso-8601=seconds)"
