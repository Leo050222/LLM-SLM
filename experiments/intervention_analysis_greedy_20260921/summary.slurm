#!/bin/bash
#SBATCH -J analysis_greedy_summary
#SBATCH -p yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=UNLIMITED
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/analysis_greedy_summary.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/analysis_greedy_summary.%j.err
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
TASK_SCRIPT_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/intervention_analysis_greedy_20260921
TASK_PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
for script in validate.py summarize.py compare_tokens.py continuous_support.py validate_mass.py teacher_uncertainty.py compare_regimes.py; do
 "$TASK_PYTHON" -u "$TASK_SCRIPT_ROOT/$script"
done
