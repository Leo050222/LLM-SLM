#!/bin/bash
#SBATCH -J verify_intervention_logits
#SBATCH -p yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=UNLIMITED
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/intervention_logits_verify.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/intervention_logits_verify.%j.err
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python \
 /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/intervention_logits_bf16_x8_20260920/run.py verify
