#!/bin/bash
#SBATCH -J intervention_logits_bf16
#SBATCH -p yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H800:1
#SBATCH --mem=96G
#SBATCH --exclude=gnho005,gnho009,gnho018
#SBATCH --time=UNLIMITED
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/intervention_logits_bf16.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/intervention_logits_bf16.%j.err
set -Eeuo pipefail
ROLE=${1:?student or teacher required}
case "$ROLE" in student|teacher) ;; *) exit 2 ;; esac
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
/soft/slurm/bin/srun --nodes=1 --ntasks=1 --cpus-per-task=8 --cpu-bind=cores --kill-on-bad-exit=1 \
 /storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python \
 "$ROOT/opd_scripts/intervention_logits_bf16_x8_20260920/run.py" score --role "$ROLE"
