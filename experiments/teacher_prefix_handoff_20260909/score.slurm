#!/bin/bash
#SBATCH --job-name=prefix_cpu_score
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/prefix_score.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/prefix_score.%j.err
set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
export PYTHONPATH=${ROOT}/opd_scripts:${ROOT}/verl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python "${ROOT}/opd_scripts/teacher_prefix_handoff_20260909/score.py"
