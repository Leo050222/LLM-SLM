#!/bin/bash
#SBATCH --job-name=dm_ge6_train25
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_ge6_train25.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_ge6_train25.%j.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd
export OMP_NUM_THREADS=8

mkdir -p /storage/yukaichengLab/wangxin/test_eiloeo/opd_logs
python /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/build_deepmath_ge6_train25pct_perdifficulty.py
