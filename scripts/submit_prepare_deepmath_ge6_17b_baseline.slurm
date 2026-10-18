#!/bin/bash
#SBATCH --job-name=dm_ge6_17prep
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_ge6_17prep.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_ge6_17prep.%j.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
python ${ROOT}/opd_scripts/prepare_deepmath_ge6_eval_shards.py \
  --input ${ROOT}/opd_data/deepmath_ge6_opd_train15pct_seed20260811/test_1000.parquet \
  --output-dir ${ROOT}/opd_data/deepmath_ge6_opd_train15pct_seed20260811/eval_shards4 \
  --expected-sha256 6ef8e66d181b7522bad2904ad1fcf20d8fc152af0c9d7eb05e07a9f2f0415ae0 \
  --num-shards 4
