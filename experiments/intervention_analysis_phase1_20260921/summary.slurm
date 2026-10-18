#!/bin/bash
#SBATCH -J intervention_phase1_summary
#SBATCH -p yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=UNLIMITED
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/intervention_phase1_summary.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/intervention_phase1_summary.%j.err
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python -u /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/intervention_analysis_phase1_20260921/summarize.py
