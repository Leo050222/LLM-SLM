#!/bin/bash
#SBATCH -J crux_logits_replay
#SBATCH -p yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H800:1
#SBATCH --mem=48G
#SBATCH --exclude=gnho009,gnho018
#SBATCH --time=00:20:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/logits_replay.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/logits_replay.%j.err
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nvidia-smi --query-gpu=index,uuid,name,memory.used --format=csv
/soft/slurm/bin/srun --nodes=1 --ntasks=1 --cpus-per-task=8 /storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/logits_replay_20260918/run.py
