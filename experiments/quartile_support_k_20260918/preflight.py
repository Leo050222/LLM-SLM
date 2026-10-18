import json,os,sys,subprocess
from pathlib import Path
from collections import Counter
from ablation_core import atomic_json
from branch import capacity
from code_scorer import sandbox
from prepare import P,R
from math_rules import score_answer
rows=list(map(json.loads,(P/'prompts.jsonl').read_text().splitlines()))
assert min(capacity(r,r['anchor']['response_prefix_token_ids']) for r in rows)>=32,'A sampled prefix cannot fit the requested 32-token bridge'
counts=Counter(r['dataset'] for r in rows);assert len(counts)==8 and set(counts.values())=={100}
for ds in ['bbh','math500','olympiad_bench']:
 row=next(r for r in rows if r['dataset']==ds);x=json.loads(Path(row['anchor']['source']).read_text())
 result=subprocess.run([sys.executable,str(Path(__file__).parent/'score.py'),'--math-one'],input=json.dumps(x),capture_output=True,text=True,timeout=120,check=True)
 assert any(l.startswith('MATH_RESULT=') for l in result.stdout.splitlines()),result.stdout
 print('MATH_SCORER_SMOKE',ds,flush=True)
p=json.loads((R/'opd_eval_code/code_reasoning_grid_20260913/data/HumanEvalPlus.jsonl').read_text().splitlines()[0])
result=sandbox(dict(dataset='humaneval_plus',problem=p,solution=p['prompt']+p['canonical_solution']));assert result['correct'],result
atomic_json(P/'PREFLIGHT_PASSED.json',dict(rows=800,all_bridge_requests_32=True,math_scorers=3,code_scorer=True))
print('PREFLIGHT_PASSED',flush=True)
