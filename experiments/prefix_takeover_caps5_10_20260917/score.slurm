#!/bin/bash
#SBATCH -J prefix_caps5_10_score
#SBATCH -p yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/prefix_caps5_10_score.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/prefix_caps5_10_score.%j.err
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
S=/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/prefix_takeover_caps5_10_20260917
P=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_prefix/prefix_takeover_caps5_10_20260917
PY=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
test -f "$P/judge/EXTERNAL_SCORING_APPROVED.json"
PORT=$($PY -c "import json;print(json.load(open('$P/judge/relay.json'))['port'])")
$PY "$S/judge_client.py" --tunnel-port "$PORT" --probe
$PY "$S/score.py"
