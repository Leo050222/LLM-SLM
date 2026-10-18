import copy,unittest
from ablation_core import Settings,digest
from generation import generate
from deadline import TraceDeadline
class RPC:
 def __init__(self):self.calls=[]
 def call(self,role,payload):
  self.calls.append((role,copy.deepcopy(payload)))
  return dict(token_ids=[42],finish_reason='stop' if len(self.calls)>=2 else 'length')
row=dict(dataset_index=7,prompt_token_ids=[1,2,3],anchor={'response_prefix_token_ids':[999]})
model=dict(max_model_len=40960,model='test')
class Tests(unittest.TestCase):
 def test_original_prompt_role_seed_and_no_ban(self):
  for role in ['student','teacher']:
   s=Settings(mode=role,seed=20260911);rpc=RPC();out=generate(row,s,model,rpc)
   self.assertEqual(rpc.calls[0][1]['prefix'],[1,2,3]);self.assertEqual(out['token_sources'],[role,role])
   self.assertTrue(all(r==role and p['op']=='generate' and 'banned_first' not in p for r,p in rpc.calls))
   self.assertEqual(rpc.calls[0][1]['seed'],int(digest([20260911,7,0])[:8],16)%(2**31));self.assertEqual(out['events'],[])
 def test_resume_seed_and_prefix(self):
  class Stop(Exception):pass
  saved=[]
  def save(x):saved.append(copy.deepcopy(x));raise Stop()
  s=Settings(mode='teacher',seed=20260912)
  with self.assertRaises(Stop):generate(row,s,model,RPC(),save=save)
  rpc=RPC();generate(row,s,model,rpc,saved[0]);self.assertEqual(rpc.calls[0][1]['prefix'],[1,2,3,42]);self.assertEqual(rpc.calls[0][1]['seed'],int(digest([20260912,7,1])[:8],16)%(2**31))
 def test_context_and_budget_error(self):
  s=Settings(mode='student');rpc=RPC();out=generate(row,s,dict(model,max_model_len=4),rpc);self.assertEqual(out['finish_reason'],'context_limit');self.assertFalse(rpc.calls)
  class DeadlineRPC:
   def call(self,*args):raise TraceDeadline()
  with self.assertRaises(TraceDeadline):generate(row,s,model,DeadlineRPC())
 def test_checkpoint_identity(self):
  s=Settings(mode='student');x=generate(row,s,model,RPC())
  with self.assertRaises(AssertionError):generate(row,Settings(mode='teacher'),model,RPC(),x)
if __name__=='__main__':unittest.main()
