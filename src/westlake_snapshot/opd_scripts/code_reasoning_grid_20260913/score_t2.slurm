#!/bin/bash
#SBATCH -J code_score_t2
#SBATCH -p yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/code_score_t2.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/code_score_t2.%j.err
set -Eeuo pipefail
export SCORE_GROUPS=k1_t2,k4_t2,k8_t2,k16_t2,k32_t2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/code_reasoning_grid_20260913/score.py
