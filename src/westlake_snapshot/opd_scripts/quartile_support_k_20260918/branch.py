from ablation_core import digest
KS=(1,4,8,16,32)
MAX_TOKENS=40264
MAX_MODEL_LEN=40960

def capacity(row,response):
 return min(MAX_TOKENS-len(response),MAX_MODEL_LEN-len(row['prompt_token_ids'])-len(response))

def make_bridge(row,call):
 prefix=row['anchor']['response_prefix_token_ids'];n=32;assert capacity(row,prefix)>=32
 out=call('teacher',row['prompt_token_ids']+prefix,n,1.,0)
 assert 0<len(out['token_ids'])<=n
 assert out['finish_reason']=='stop' or len(out['token_ids'])==n
 return dict(identity=digest(row),token_ids=out['token_ids'],finish_reason=out['finish_reason'],requested_tokens=n,temperature=1.,seed_namespace='teacher_bridge_once')

def run_branch(row,k,bridge,call,state=None,save=lambda s:None):
 assert k in KS and bridge['identity']==digest(row)
 fixed=row['anchor']['response_prefix_token_ids'];take=bridge['token_ids'][:k]
 identity=digest(dict(row=row,k=k,bridge=bridge,student_temperature=0.,limits=[MAX_TOKENS,MAX_MODEL_LEN]))
 if state is None:
  state=dict(identity=identity,status='RUNNING',response_token_ids=fixed+take,actual_teacher_tokens=len(take),student_rounds=0)
  if bridge['finish_reason']=='stop' and k>=len(bridge['token_ids']):state.update(status='COMPLETE',finish_reason='teacher_stop')
  save(state)
 assert state['identity']==identity
 assert state['response_token_ids'][:len(fixed)+len(take)]==fixed+take
 while state['status']!='COMPLETE':
  n=min(256,capacity(row,state['response_token_ids']))
  if n<=0:state.update(status='COMPLETE',finish_reason='length_limit');save(state);break
  out=call('student',row['prompt_token_ids']+state['response_token_ids'],n,0.,state['student_rounds'])
  assert len(out['token_ids'])<=n and (out['token_ids'] or out['finish_reason']=='stop')
  state['response_token_ids']+=out['token_ids'];state['student_rounds']+=1
  if out['finish_reason']=='stop':state.update(status='COMPLETE',finish_reason='stop')
  save(state)
 return state

def verify(row,k,bridge,state):
 fixed=row['anchor']['response_prefix_token_ids'];take=bridge['token_ids'][:k]
 assert state['status']=='COMPLETE' and state['response_token_ids'][:len(fixed)+len(take)]==fixed+take
 assert state['actual_teacher_tokens']==len(take)
 assert state['identity']==digest(dict(row=row,k=k,bridge=bridge,student_temperature=0.,limits=[MAX_TOKENS,MAX_MODEL_LEN]))
 assert len(state['response_token_ids'])<=MAX_TOKENS
