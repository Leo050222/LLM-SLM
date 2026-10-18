#!/bin/bash
#SBATCH -J teacher_only100x8_score
#SBATCH -p yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=UNLIMITED
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/teacher_only100x8_score.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/teacher_only100x8_score.%j.err
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
S=/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/teacher_only_sample100_x8_20260917
P=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_prefix/teacher_only_sample100_x8_20260917
PY=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
test -f "$P/judge/EXTERNAL_SCORING_APPROVED.json"
PORT=$($PY -c "import json;print(json.load(open('$P/judge/relay.json'))['port'])")
$PY "$S/judge_client.py" --tunnel-port "$PORT" --probe
$PY "$S/score.py"
