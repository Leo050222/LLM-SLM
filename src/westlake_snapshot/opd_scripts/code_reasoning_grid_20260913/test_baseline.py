import ast, unittest, json, hashlib
from dataclasses import asdict
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'nojudge_grid_20260912'))
from ablation_core import Settings,digest
module=ast.parse(Path(__file__).with_name('dispatch.py').read_text())
ns=dict(asdict=asdict,digest=digest)
exec(compile(ast.Module(body=[x for x in module.body if isinstance(x,ast.FunctionDef) and x.name=='baseline_rollout'],type_ignores=[]),'baseline','exec'),ns)
f=ns['baseline_rollout']
class TestBaseline(unittest.TestCase):
 def test_roles_stop_resume(self):
  for role in ['student','teacher']:
   row=dict(dataset_index=0,prompt_token_ids=[10]); settings=Settings(max_tokens=5,chunk_tokens=2)
   state=dict(identity=digest(dict(row=row,settings=asdict(settings))),status='RUNNING',response_token_ids=[],token_sources=[],rounds=0,student_draft_tokens=0)
   class Backend:
    max_model_len=100
    calls=[]
    def call(self,r,**kw):
     self.calls.append((r,kw));return dict(token_ids=[1],finish_reason='stop')
   b=Backend();b.calls=[]
   result=f(row,settings,b,state,lambda x:None,role)
   self.assertEqual(result['token_sources'],[role]);self.assertEqual(result['finish_reason'],'stop')
   self.assertEqual(b.calls[0][0],role);self.assertNotIn('banned_first',b.calls[0][1])
   f(row,settings,b,state,lambda x:None,role);self.assertEqual(len(b.calls),1)
 def test_budget_limit(self):
  row=dict(dataset_index=0,prompt_token_ids=[10]);s=Settings(max_tokens=3,chunk_tokens=2)
  state=dict(identity=digest(dict(row=row,settings=asdict(s))),status='RUNNING',response_token_ids=[],token_sources=[],rounds=0,student_draft_tokens=0)
  class B:
   max_model_len=100
   def call(self,r,**kw):return dict(token_ids=[1]*kw['n'],finish_reason='length')
  self.assertEqual(len(f(row,s,B(),state,lambda x:None,'student')['response_token_ids']),3)
if __name__=='__main__':unittest.main()
