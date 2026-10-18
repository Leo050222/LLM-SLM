#!/bin/bash
#SBATCH --job-name=dm_ge6_native_nt_merge
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-1
#SBATCH --time=01:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_ge6_native_nt_merge.%A_%a.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_ge6_native_nt_merge.%A_%a.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
MODEL_LABELS=(qwen3_1_7b qwen3_14b)
MODELS=(/storage/yukaichengLab/share/model/Qwen3-1.7B /storage/yukaichengLab/share/model/Qwen3-14B)
MODEL_LABEL=${MODEL_LABELS[${SLURM_ARRAY_TASK_ID}]}
MODEL=${MODELS[${SLURM_ARRAY_TASK_ID}]}
RUN=${ROOT}/opd_eval_deepmath/ge6_test1000_native_nonthinking_seed20260811/${MODEL_LABEL}

python ${ROOT}/opd_scripts/merge_deepmath_ge6_native_nonthinking.py \
  --shard-root ${RUN}/shards --output-dir ${RUN}/merged \
  --full-data ${ROOT}/opd_data/deepmath_ge6_opd_train15pct_seed20260811/test_1000.parquet \
  --expected-data-sha256 6ef8e66d181b7522bad2904ad1fcf20d8fc152af0c9d7eb05e07a9f2f0415ae0 \
  --expected-model ${MODEL} --expected-tokenizer /storage/yukaichengLab/share/model/Qwen3-1.7B \
  --expected-seed 20260811 --expected-max-tokens 40264 --expected-max-model-len 40960 \
  --model-label ${MODEL_LABEL}
