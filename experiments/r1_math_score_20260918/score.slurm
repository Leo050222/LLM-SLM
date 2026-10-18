#!/bin/bash
#SBATCH -J r1_math_score
#SBATCH -p yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=UNLIMITED
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/r1_math_score.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/r1_math_score.%j.err
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
S=/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/r1_math_score_20260918
P=/storage/yukaichengLab/wangxin/test_eiloeo/opd_rescore/r1_math_score_20260918
PY=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
test -f "$P/judge/EXTERNAL_SCORING_APPROVED.json"
PORT=$($PY -c "import json;print(json.load(open('$P/judge/relay.json'))['port'])")
$PY "$S/judge_client.py" --tunnel-port "$PORT" --probe
$PY "$S/score.py"
