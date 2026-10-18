#!/bin/bash
#SBATCH -J strong_reprompt_score
#SBATCH -p yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/strong_reprompt_score_20261005.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/strong_reprompt_score_20261005.%j.err
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
S=/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/strong_reprompt_score_20261005
P=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_ablation/strong_reprompt_score_20261005
PY=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
test -f "$P/judge/EXTERNAL_SCORING_APPROVED.json"
test -f "$P/judge/relay.json"
$PY "$S/judge_client.py" --tunnel-port "$($PY -c "import json;print(json.load(open('$P/judge/relay.json'))['port'])")" --probe
$PY "$S/score.py"
