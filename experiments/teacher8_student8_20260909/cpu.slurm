#!/bin/bash
#SBATCH --job-name=t8s8_cpu_score
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/t8s8_score.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/t8s8_score.%j.err
set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
export PYTHONPATH=${ROOT}/opd_scripts:${ROOT}/verl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python "${ROOT}/opd_scripts/teacher8_student8_20260909/score.py" --output "${1:-${ROOT}/opd_eval_deepmath/teacher8_student8_20260909}" --limit "${2:-1000}" --workers "${3:-32}"
