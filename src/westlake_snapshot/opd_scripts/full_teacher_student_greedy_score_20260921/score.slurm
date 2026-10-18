#!/bin/bash
#SBATCH -J full_greedy_score
#SBATCH -p yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/full_greedy_score.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/full_greedy_score.%j.err
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
S=/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/full_teacher_student_greedy_score_20260921
P=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_baselines/full_teacher_student_greedy_20260920
PY=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
test -f "$P/ALL_GENERATION_COMPLETE.json"
test -f "$P/judge/EXTERNAL_SCORING_APPROVED.json"
PORT=$($PY -c "import json;print(json.load(open('$P/judge/relay.json'))['port'])")
$PY "$S/judge_client.py" --tunnel-port "$PORT" --probe
$PY "$S/score.py"
