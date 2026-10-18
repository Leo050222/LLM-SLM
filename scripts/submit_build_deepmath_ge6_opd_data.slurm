#!/bin/bash
#SBATCH --job-name=dm_ge6_data
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_ge6_data.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_ge6_data.%j.err

set -euo pipefail

source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=16

mkdir -p /storage/yukaichengLab/wangxin/test_eiloeo/opd_logs
python /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/build_deepmath_ge6_opd_data.py
