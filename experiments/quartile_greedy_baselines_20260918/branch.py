from ablation_core import digest
MAX_TOKENS=40264
MAX_MODEL_LEN=40960
def identity(row,role):return digest(dict(row=row,role=role,student_temperature=0.,limits=[MAX_TOKENS,MAX_MODEL_LEN]))
def run_branch(row,call,state=None,save=lambda s:None,role="student"):
 assert role in ("student","teacher")
 fixed=row['anchor']['response_prefix_token_ids']
 if state is None:state=dict(identity=identity(row,role),status='RUNNING',response_token_ids=list(fixed),actual_teacher_tokens=0,student_rounds=0)
 assert state['identity']==identity(row,role) and state['response_token_ids'][:len(fixed)]==fixed
 while state['status']!='COMPLETE':
  n=min(256,MAX_TOKENS-len(state['response_token_ids']),MAX_MODEL_LEN-len(row['prompt_token_ids'])-len(state['response_token_ids']))
  if n<=0:state.update(status='COMPLETE',finish_reason='length_limit');save(state);break
  out=call(role,row['prompt_token_ids']+state['response_token_ids'],n,0.,state['student_rounds'])
  assert len(out['token_ids'])<=n and (out['token_ids'] or out['finish_reason']=='stop')
  state['response_token_ids']+=out['token_ids'];state['student_rounds']+=1
  state['actual_teacher_tokens']=(len(state['response_token_ids'])-len(fixed)) if role=='teacher' else 0
  if out['finish_reason']=='stop':state.update(status='COMPLETE',finish_reason='stop')
  save(state)
 return state
def verify(row,state,role="student"):
 fixed=row['anchor']['response_prefix_token_ids']
 assert state['status']=='COMPLETE' and state['response_token_ids'][:len(fixed)]==fixed
 assert state['actual_teacher_tokens']==((len(state['response_token_ids'])-len(fixed)) if role=='teacher' else 0) and state['identity']==identity(row,role)
 assert len(state['response_token_ids'])<=MAX_TOKENS
