#!/bin/bash
#SBATCH --job-name=dm_ge6_ckptmerge
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-3
#SBATCH --time=01:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_ge6_ckptmerge.%A_%a.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_ge6_ckptmerge.%A_%a.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
STEPS=(27 80 106 133)
STEP=${STEPS[${SLURM_ARRAY_TASK_ID}]}
RUN=${ROOT}/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_top16_rkl_step${STEP}_seed20260811
BASELINE=${ROOT}/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_baseline_seed20260811/merged/rollouts.jsonl
MODEL=${ROOT}/opd_checkpoints/qwen3_1_7b_from_14b_deepmath_ge6_train15pct_student_top16_rkl_16h800/global_step_${STEP}/actor/huggingface

if [[ -e "${RUN}/merged/STEP${STEP}_EVAL_COMPLETE" ]]; then
  echo "Refusing to overwrite completed merged evaluation: ${RUN}/merged" >&2
  exit 3
fi

python ${ROOT}/opd_scripts/merge_deepmath_ge6_step53_eval.py \
  --shard-root ${RUN}/shards --output-dir ${RUN}/merged \
  --full-data ${ROOT}/opd_data/deepmath_ge6_opd_train15pct_seed20260811/test_1000.parquet \
  --baseline-rollouts ${BASELINE} \
  --expected-data-sha256 6ef8e66d181b7522bad2904ad1fcf20d8fc152af0c9d7eb05e07a9f2f0415ae0 \
  --expected-model ${MODEL} \
  --expected-tokenizer /storage/yukaichengLab/share/model/Qwen3-1.7B \
  --expected-seed 20260811 --expected-max-tokens 40264 --expected-max-model-len 40960 \
  --checkpoint-step ${STEP}
