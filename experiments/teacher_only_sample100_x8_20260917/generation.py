from ablation_core import atomic_json,digest

def identity(row,seed,model):
 return digest(dict(row=row,seed=seed,model=model,condition='teacher_only',start='original_prompt',max_tokens=40264,chunk_tokens=64))

def generate(row,seed,model,rpc,state=None,save=lambda s:None):
 ident=identity(row,seed,model)
 if state is None:state=dict(identity=ident,response_token_ids=[],token_sources=[],rounds=0,status='RUNNING')
 assert state['identity']==ident
 if state['status']=='COMPLETE':return state
 while len(state['response_token_ids'])<40264:
  prefix=row['prompt_token_ids']+state['response_token_ids']
  n=min(64,40264-len(state['response_token_ids']),model['max_model_len']-len(prefix)-1)
  if n<=0:state['finish_reason']='context_limit';break
  step_seed=int(digest([seed,row['dataset_index'],state['rounds']])[:8],16)%(2**31)
  out=rpc.call('teacher',dict(model_identity=digest(model),op='generate',prefix=prefix,n=n,seed=step_seed))
  ids=out['token_ids'];assert 0<len(ids)<=n
  state['response_token_ids']+=ids;state['token_sources']+=['teacher']*len(ids);state['rounds']+=1
  if out['finish_reason']=='stop':state.update(status='COMPLETE',finish_reason='stop')
  save(state)
  if state['status']=='COMPLETE':return state
 else:state['finish_reason']='length'
 state['status']='COMPLETE';save(state);return state

def verify(row,seed,model,x):
 assert x['identity']==identity(row,seed,model) and x['status']=='COMPLETE'
 assert 0<len(x['response_token_ids'])<=40264
 assert x['token_sources']==['teacher']*len(x['response_token_ids'])
 assert x['finish_reason'] in ['stop','length','context_limit']
 if x['finish_reason']=='stop':assert x['response_token_ids'][-1]==model['eos_token_id']
