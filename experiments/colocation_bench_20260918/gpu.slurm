#!/bin/bash
#SBATCH -J coloc_14b_1p7b_bench
#SBATCH -p yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:H800:2
#SBATCH --mem=192G
#SBATCH --exclude=gnho009,gnho018
#SBATCH --time=UNLIMITED
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/colocation_bench.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/colocation_bench.%j.err
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export BENCH_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_profile/colocation_bench_20260918
S=/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/colocation_bench_20260918
PY=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
/soft/slurm/bin/srun --nodes=1 --ntasks=1 --cpus-per-task=32 --cpu-bind=cores "$PY" "$S/launch.py"
