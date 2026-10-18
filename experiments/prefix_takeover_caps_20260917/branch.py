from dataclasses import asdict
from ablation_core import digest,rollout

def initial(row,settings):
 prefix=row['anchor']['response_prefix_token_ids']
 return dict(identity=digest({'row':row,'settings':asdict(settings)}),response_token_ids=list(prefix),token_sources=['student']*len(prefix),events=[],rounds=0,student_draft_tokens=0,discarded_tokens=0,judge_calls=0,status='RUNNING',initial_anchor_applied=False)

def run_branch(row,settings,backend,checkpoint=None,save=lambda x:None):
 state=checkpoint if checkpoint is not None else initial(row,settings)
 assert state['identity']==digest({'row':row,'settings':asdict(settings)})
 if state['status']=='COMPLETE':return state
 anchor=row['anchor'];fixed=anchor['response_prefix_token_ids']
 assert state['response_token_ids'][:len(fixed)]==fixed
 if settings.max_takeovers>0 and not state['initial_anchor_applied']:
  assert state['response_token_ids']==fixed and not state['events']
  prefix=row['prompt_token_ids']+fixed
  assert digest(prefix)==anchor['frozen_prefix_sha256']
  assert len(fixed)<settings.max_tokens and len(prefix)<backend.max_model_len-1
  seed=int(digest([settings.seed,row['dataset_index'],'initial_teacher'])[:8],16)%(2**31)
  out=backend.generate('teacher',prefix,1,seed,banned_first=anchor['entry_token'])
  assert len(out['token_ids'])==1 and out['token_ids'][0]!=anchor['entry_token']
  state['response_token_ids']+=out['token_ids'];state['token_sources'].append('teacher')
  state['events'].append(dict(round=-1,response_offset=len(fixed),draft_tokens=0,judgments=[],anchor_response_position=len(fixed),bridge_token_ids=out['token_ids'],requested_teacher_tokens=1,selected_candidate=anchor['candidate'],student_entry_token=anchor['entry_token'],first_token_ban=True,discarded_tokens=0,forced_initial_anchor=True,anchor_provenance=anchor['source_sha256']))
  state['initial_anchor_applied']=True
  if out['finish_reason']=='stop':state.update(status='COMPLETE',finish_reason='stop')
  save(state)
  if state['status']=='COMPLETE':return state
 return rollout(row,settings,backend,None,state,save)

def verify(row,cap,state):
 fixed=row['anchor']['response_prefix_token_ids'];tokens=state['response_token_ids']
 assert state['status']=='COMPLETE' and tokens[:len(fixed)]==fixed
 assert state['token_sources'][:len(fixed)]==['student']*len(fixed)
 assert len(state['token_sources'])==len(tokens) and state['judge_calls']==0
 events=[e for e in state['events'] if 'bridge_token_ids' in e]
 assert len(events)<=cap and state['token_sources'].count('teacher')==len(events)
 if cap==0:assert not state['initial_anchor_applied'] and not events
 else:assert state['initial_anchor_applied'] and events[0]['anchor_response_position']==len(fixed) and events[0]['forced_initial_anchor']
 for e in events:
  assert len(e['bridge_token_ids'])==1 and e['bridge_token_ids'][0]!=e['student_entry_token']
  assert e['selected_candidate']['signed_logprob_gap_student_minus_teacher']>2 and not e['judgments']
 if len(events)==cap and cap:
  assert all(t=='student' for t in state['token_sources'][events[-1]['anchor_response_position']+1:])
 return len(events)
