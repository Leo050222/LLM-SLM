#!/bin/bash
#SBATCH --job-name=dm_max_14merge
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_max_14merge.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_max_14merge.%j.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
RUN=${ROOT}/opd_eval_deepmath/maxlen_seed20260809
python ${ROOT}/opd_scripts/merge_deepmath_teacher_rollouts.py \
  --shard-root ${RUN}/qwen3_14b_shards --output-dir ${RUN}/qwen3_14b \
  --expected-max-tokens 40512 --expected-max-model-len 40960 \
  --full-data ${ROOT}/opd_data/deepmath_103k/sample_1000.parquet

