import json
import os
from pathlib import Path
from ablation_core import atomic_json
from code_scorer import sandbox

R = Path('/storage/yukaichengLab/wangxin/test_eiloeo')
p = json.loads((R/'opd_eval_code/code_reasoning_grid_20260913/data/cruxeval.jsonl').read_text().splitlines()[0])
good = sandbox(dict(dataset='cruxeval_i', problem=p, solution=p['input']))
bad = sandbox(dict(dataset='cruxeval_i', problem=p, solution='__opd_undefined_input__'))
assert good['correct'] is True and bad['correct'] is False, (good, bad)
atomic_json(Path(os.environ['OPD_ONLINE_RUN'])/'SCORING_SMOKE.json', dict(passed=True, correct_reference=good, invalid_input=bad))
print('SCORING_SMOKE_PASSED', flush=True)
