import json
from pathlib import Path
from score import sandbox
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');D=R/'opd_eval_code/code_reasoning_grid_20260913/data'
cr=json.loads((D/'cruxeval.jsonl').read_text().splitlines()[0])
for ds,sol in [('cruxeval_o',cr['output']),('cruxeval_i',cr['input'])]:
 for valid in [True,False]:
  res=sandbox(dict(dataset=ds,problem=cr,solution=sol if valid else 'None'));print(ds,valid,res,flush=True);assert res['correct']==valid
for ds,name in [('humaneval_plus','HumanEvalPlus'),('mbpp_plus','MbppPlus')]:
 p=json.loads((D/(name+'.jsonl')).read_text().splitlines()[0])
 for valid in [True,False]:
  sol=p['prompt']+p['canonical_solution'] if valid else 'def '+p['entry_point']+'(*args): return None'
  res=sandbox(dict(dataset=ds,problem=p,solution=sol));print(ds,valid,res,flush=True);assert res['correct']==valid
p=dict(question_title='smoke',question_content='echo',platform='leetcode',question_id='smoke',contest_id='0',contest_date='2025-01-01',starter_code='class Solution:',difficulty='easy',public_test_cases=json.dumps([dict(input='1',output='2',testtype='functional')]),private_test_cases='[]',metadata=json.dumps(dict(func_name='add')))
for valid in [True,False]:
 sol='class Solution:\n def add(self, x): return '+('x+1' if valid else '0')
 res=sandbox(dict(dataset='livecodebench',problem=p,solution=sol));print('livecodebench',valid,res,flush=True);assert res['correct']==valid
(Path(__file__).parent/'SCORE_SMOKE_PASS').write_text('All 5 positive and negative witnesses passed')
