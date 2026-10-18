#!/bin/bash
#SBATCH --job-name=verl_env_test
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H800:1
#SBATCH --time=00:10:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/verl_env_test.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/verl_env_test.%j.err

set -euo pipefail

source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

export PYTHONPATH=/storage/yukaichengLab/wangxin/test_eiloeo/verl
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

python /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/validate_verl_opd_env.py
