from threshold_core import rollout,Settings
class B:
 max_model_len=40960
 def __init__(self,trigger):self.trigger=trigger
 def generate(self,role,prefix,n,seed,banned_first=None):
  if role=='teacher':return dict(token_ids=[9]*n,finish_reason='stop' if len(prefix)>150 else 'length')
  return dict(token_ids=[1]*min(n,16),finish_reason='stop' if not self.trigger else 'length')
 def candidates(self,prefix,ids,threshold):
  return [dict(response_position=x,student_token_id=1,absolute_logprob_gap=3.) for x in [2,5]] if self.trigger else []
def forbidden(*a):raise AssertionError('unexpected judge')
r=dict(dataset_index=0,prompt_token_ids=[0],problem='test')
v=rollout(r,Settings(max_tokens=300),B(True),forbidden)
e=[x for x in v['events'] if 'bridge_token_ids' in x]
assert len(e)>8 and all(x['threshold_candidate']['response_position']==2 for x in e)
assert v['judge_calls']==0 and all(not x['judgments'] for x in v['events'])
assert all(len(x['bridge_token_ids'])==8 for x in e)
assert rollout(r,Settings(max_tokens=300),B(True),forbidden,v)==v
v=rollout(r,Settings(),B(False),forbidden);assert v['status']=='COMPLETE' and all(x=='student' for x in v['token_sources'])
print('first-threshold selection, no judge, unlimited takeovers, bridge8, no-candidate and resume passed')
