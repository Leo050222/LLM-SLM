#!/bin/bash
#SBATCH -J math_score_finalize
#SBATCH -p yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=UNLIMITED
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/math_score_finalize.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/math_score_finalize.%j.err
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
S=/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/math_grid_score_all_2000_20260919
P=/storage/yukaichengLab/wangxin/test_eiloeo/opd_rescore/math_grid_score_all_2000_20260919
PY=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
test -f "$P/judge/EXTERNAL_SCORING_APPROVED.json"
$PY "$S/repair_finalize.py"
