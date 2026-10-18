#!/bin/bash
#SBATCH -J code_score
#SBATCH -p yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/code_score.%A_%a.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/code_score.%A_%a.err
set -Eeuo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/code_reasoning_grid_20260913/score.py
