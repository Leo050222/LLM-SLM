import json,os,subprocess,sys
from pathlib import Path
from ablation_core import atomic_json
from code_scorer import sandbox
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');P=Path(os.environ['GRID_ROOT']);slot=Path(os.environ['OPD_ONLINE_RUN'])
p=json.loads((R/'opd_eval_code/code_reasoning_grid_20260913/data/HumanEvalPlus.jsonl').read_text().splitlines()[0])
result=sandbox({'dataset':'humaneval_plus','problem':p,'solution':p['prompt']+p['canonical_solution']})
assert result['correct'],result
# Import all math extractors before allowing a long generation run.
import score
atomic_json(slot/'SCORING_SMOKE.json',{'passed':True,'humaneval_plus':result})
print('SCORING_SMOKE_PASSED',flush=True)
