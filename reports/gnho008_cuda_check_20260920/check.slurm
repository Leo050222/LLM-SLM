#!/bin/bash
#SBATCH -J gnho008_cuda_check
#SBATCH -p yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodelist=gnho008
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H800:8
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/gnho008_cuda_check.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/gnho008_cuda_check.%j.err
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/soft/slurm/bin/srun --ntasks=1 --cpus-per-task=8 /storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/gnho008_cuda_check_20260920/check.py
