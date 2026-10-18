#!/bin/bash
#SBATCH -J dynamic_k_score
#SBATCH -p yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dynamic_k_score.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dynamic_k_score.%j.err
set -Eeuo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python -u /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/dynamic_k_entropy_20260914/score_dynamic.py
