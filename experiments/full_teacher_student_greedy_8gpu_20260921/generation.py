from dataclasses import asdict
from ablation_core import digest

def generate(row,settings,model,rpc,state=None,save=lambda s:None):
 role=settings.mode;assert role in ('student','teacher')
 assert settings.temperature==0.0
 ident=digest(dict(row=row,settings=asdict(settings)))
 if state is None:state=dict(identity=ident,response_token_ids=[],token_sources=[],events=[],rounds=0,student_draft_tokens=0,discarded_tokens=0,judge_calls=0,status='RUNNING')
 assert state['identity']==ident
 if state['status']=='COMPLETE':return state
 while len(state['response_token_ids'])<settings.max_tokens:
  prefix=row['prompt_token_ids']+state['response_token_ids']
  n=min(settings.chunk_tokens,settings.max_tokens-len(state['response_token_ids']),model['max_model_len']-len(prefix)-1)
  if n<=0:state['finish_reason']='context_limit';break
  seed=int(digest([settings.seed,row['dataset_index'],state['rounds']])[:8],16)%(2**31)
  out=rpc.call(role,dict(model_identity=digest(model),op='generate',prefix=prefix,n=n,seed=seed))
  ids=out['token_ids'];assert 0<len(ids)<=n
  state['response_token_ids']+=ids;state['token_sources'] += [role]*len(ids);state['rounds']+=1
  if role=='student':state['student_draft_tokens']+=len(ids)
  save(state)
  if out['finish_reason']=='stop':state['finish_reason']='stop';break
 else:state['finish_reason']='length'
 state['status']='COMPLETE';save(state);return state

def verify(row,settings,x):
 assert x['identity']==digest(dict(row=row,settings=asdict(settings)))
 assert x['status']=='COMPLETE' and x['judge_calls']==0 and not x['events']
 assert x['token_sources']==[settings.mode]*len(x['response_token_ids'])
 assert x['prompt_token_ids']==row['prompt_token_ids'] and x['problem_id']==row['problem_id']
 assert len(x['response_token_ids'])<=settings.max_tokens
 assert x['finish_reason'] in ('stop','length','context_limit','trace_timeout')
 if x['finish_reason']=='trace_timeout':assert x['timeout_count_as_wrong'] and x['trace_elapsed_seconds']>=900
