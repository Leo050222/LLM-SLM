#!/bin/bash
#SBATCH -J weak_reprompt_nodraft_greedy_score_20260921
#SBATCH -p yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/weak_reprompt_nodraft_greedy_score_20260921.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/weak_reprompt_nodraft_greedy_score_20260921.%j.err
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
S=/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/weak_reprompt_nodraft_greedy_score_20260921
P=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_ablation/weak_reprompt_nodraft_greedy_t2_k1_20260921
PY=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
test -f "$P/judge/EXTERNAL_SCORING_APPROVED.json"
PORT=$($PY -c "import json;print(json.load(open('$P/judge/relay.json'))['port'])")
$PY "$S/judge_client.py" --tunnel-port "$PORT" --probe
cd "$S"
$PY - <<'SMOKE'
import json
from code_scorer import sandbox,R
p=json.loads((R/'opd_eval_code/code_reasoning_grid_20260913/data/HumanEvalPlus.jsonl').read_text().splitlines()[0])
r=sandbox(dict(dataset='humaneval_plus',problem=p,solution=p['prompt']+p['canonical_solution']))
assert r['correct'],r
print('CODE_SCORING_SMOKE_PASSED',flush=True)
SMOKE
$PY "$S/score.py"
