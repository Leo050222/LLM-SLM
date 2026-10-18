#!/bin/bash
#SBATCH --job-name=dm_ge6_native_nt_cmp
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_ge6_native_nt_cmp.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_ge6_native_nt_cmp.%j.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
BASE=${ROOT}/opd_eval_deepmath/ge6_test1000_native_nonthinking_seed20260811
python ${ROOT}/opd_scripts/compare_deepmath_ge6_native_nonthinking.py \
  --qwen17-dir ${BASE}/qwen3_1_7b/merged \
  --qwen14-dir ${BASE}/qwen3_14b/merged \
  --output-dir ${BASE}/comparison
