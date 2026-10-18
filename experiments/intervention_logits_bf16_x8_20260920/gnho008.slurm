#!/bin/bash
#SBATCH -J logits_bf16_008
#SBATCH -p yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --nodelist=gnho008
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H800:2
#SBATCH --mem=192G
#SBATCH --time=UNLIMITED
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/logits_bf16_008.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/logits_bf16_008.%j.err
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
/soft/slurm/bin/srun --nodes=1 --ntasks=1 --cpus-per-task=16 --cpu-bind=cores --kill-on-bad-exit=1 \
 /storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python \
 /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/intervention_logits_bf16_x8_20260920/launch_008.py
