#!/bin/bash
#SBATCH --job-name=dm_ge6_17merge
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_ge6_17merge.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_ge6_17merge.%j.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
RUN=${ROOT}/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_baseline_seed20260811
python ${ROOT}/opd_scripts/merge_deepmath_ge6_baseline.py \
  --shard-root ${RUN}/shards --output-dir ${RUN}/merged \
  --full-data ${ROOT}/opd_data/deepmath_ge6_opd_train15pct_seed20260811/test_1000.parquet \
  --expected-data-sha256 6ef8e66d181b7522bad2904ad1fcf20d8fc152af0c9d7eb05e07a9f2f0415ae0 \
  --expected-model /storage/yukaichengLab/share/model/Qwen3-1.7B \
  --expected-seed 20260811 --expected-max-tokens 40264 --expected-max-model-len 40960
