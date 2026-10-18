#!/bin/bash
#SBATCH --job-name=dm25_nt_ckptmerge
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-4
#SBATCH --time=02:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm25_nt_ckptmerge.%A_%a.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm25_nt_ckptmerge.%A_%a.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
read -r -a STEPS <<< "${OPD_EVAL_STEPS:-44 89 133 178 222}"
STEP=${STEPS[${SLURM_ARRAY_TASK_ID}]}
CKPT_ROOT=${OPD_EVAL_CKPT_ROOT:-${ROOT}/opd_checkpoints/qwen3_1_7b_from_14b_deepmath_ge6_train25pct_nonthinking_student_top16_rkl_8h800_single_node}
EVAL_ROOT=${OPD_EVAL_ROOT:-${ROOT}/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_train25pct_nonthinking_top16_rkl_seed20260811}
MODEL=${CKPT_ROOT}/global_step_${STEP}/actor/huggingface
RUN=${EVAL_ROOT}/step_${STEP}

if [[ -e "${RUN}/merged/NONTHINKING_COMPLETE" ]]; then
  echo "Refusing to overwrite completed evaluation: ${RUN}/merged" >&2
  exit 3
fi
python ${ROOT}/opd_scripts/merge_deepmath_ge6_native_nonthinking.py \
  --shard-root ${RUN}/shards --output-dir ${RUN}/merged \
  --full-data ${ROOT}/opd_data/deepmath_ge6_opd_train15pct_seed20260811/test_1000.parquet \
  --expected-data-sha256 6ef8e66d181b7522bad2904ad1fcf20d8fc152af0c9d7eb05e07a9f2f0415ae0 \
  --expected-model ${MODEL} --expected-tokenizer /storage/yukaichengLab/share/model/Qwen3-1.7B \
  --expected-seed 20260811 --expected-max-tokens 40264 --expected-max-model-len 40960 \
  --model-label ${OPD_EVAL_MODEL_LABEL_PREFIX:-qwen3_1_7b_train25pct_nonthinking_top16_rkl}_step${STEP}
echo "DEEPMATH_TRAIN25_NONTHINKING_STEP${STEP}_MERGE_OK job=${SLURM_JOB_ID} time=$(date --iso-8601=seconds)"
