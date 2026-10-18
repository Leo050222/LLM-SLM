"""Checkpointable token-by-token dynamic teacher support, no external judge."""
from dataclasses import dataclass,asdict
from concurrent.futures import ThreadPoolExecutor
from ablation_core import Settings as BaseSettings,digest

@dataclass(frozen=True)
class Settings(BaseSettings):
    mode:str='dynamic_entropy'
    max_takeovers:int|None=None
    student_entropy_min:float=.5
    teacher_entropy_max:float=.2

def paired_entropy(backend,prefix):
    with ThreadPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(backend.call,role,op='entropy',prefix=prefix) for role in ['student','teacher']]
        s,t=[f.result() for f in fs]
    assert s['full_vocabulary'] and t['full_vocabulary']
    return s['entropy'],t['entropy']

def rollout(row,settings,backend,state=None,save=lambda x:None):
    identity=digest(dict(row=row,settings=asdict(settings)))
    state=state or dict(identity=identity,response_token_ids=[],token_sources=[],events=[],rounds=0,
                       student_draft_tokens=0,discarded_tokens=0,judge_calls=0,status='RUNNING')
    assert state['identity']==identity
    if state['status']=='COMPLETE':return state
    prompt=row['prompt_token_ids']
    def finish_bridge(reason):
        state['events'][-1]['bridge_stop_reason']=reason
        state.pop('pending_bridge',None);state['rounds']+=1;save(state)
    while True:
        if state.get('pending_bridge'):
            pending=state['pending_bridge'];e=state['events'][-1];bridge=e['bridge_token_ids']
            prefix=prompt+state['response_token_ids']
            if len(bridge)>=pending['limit']:
                finish_bridge('max8' if pending['limit']==8 else 'token_budget');continue
            if bridge:
                decisions=e['entropy_decisions']
                if not decisions or decisions[-1]['after_teacher_tokens']!=len(bridge):
                    hs,ht=paired_entropy(backend,prefix)
                    decisions.append(dict(after_teacher_tokens=len(bridge),student_entropy=hs,teacher_entropy=ht,
                        continue_teacher=hs>=settings.student_entropy_min and ht<=settings.teacher_entropy_max))
                    save(state)
                if not decisions[-1]['continue_teacher']:
                    finish_bridge('entropy_condition_false');continue
            n=len(bridge)
            out=backend.generate('teacher',prefix,1,(pending['seed']+n)%(2**31),
                                 banned_first=e['student_entry_token'] if n==0 else None)
            ids=out['token_ids'];assert len(ids)==1
            if n==0:assert ids[0]!=e['student_entry_token']
            bridge.extend(ids);state['response_token_ids'].extend(ids);state['token_sources'].append('teacher')
            save(state)
            if out['finish_reason']=='stop':
                finish_bridge('eos');state.update(status='COMPLETE',finish_reason='stop');save(state);return state
            continue
        prefix=prompt+state['response_token_ids'];offset=len(state['response_token_ids'])
        capacity=min(settings.max_tokens-offset,backend.max_model_len-len(prefix)-1)
        if capacity<=0:
            state.update(status='COMPLETE',finish_reason='length' if offset>=settings.max_tokens else 'context_limit')
            save(state);return state
        seed=int(digest([settings.seed,row['dataset_index'],state['rounds']])[:8],16)%(2**31)
        draft=backend.generate('student',prefix,min(settings.chunk_tokens,capacity),seed)
        ids=draft['token_ids'];assert 0<len(ids)<=min(settings.chunk_tokens,capacity)
        candidates=[c for c in backend.candidates(prefix,ids,settings.gap_threshold)
                    if c['signed_logprob_gap_student_minus_teacher']>settings.gap_threshold]
        assert [c['response_position'] for c in candidates]==sorted(set(c['response_position'] for c in candidates))
        e=dict(round=state['rounds'],response_offset=offset,draft_tokens=len(ids),judgments=[])
        state['student_draft_tokens']+=len(ids)
        if not candidates:
            state['response_token_ids'].extend(ids);state['token_sources'].extend(['student']*len(ids))
            state['events'].append(e);state['rounds']+=1
            if draft['finish_reason']=='stop':state.update(status='COMPLETE',finish_reason='stop')
            save(state)
            if state['status']=='COMPLETE':return state
            continue
        selected=candidates[0];pos=selected['response_position'];assert ids[pos]==selected['student_token_id']
        accepted=ids[:pos];state['response_token_ids'].extend(accepted);state['token_sources'].extend(['student']*pos)
        state['discarded_tokens']+=len(ids)-pos
        e.update(anchor_response_position=offset+pos,bridge_token_ids=[],requested_teacher_tokens=8,
                 selected_candidate=selected,student_entry_token=ids[pos],first_token_ban=True,
                 discarded_tokens=len(ids)-pos,entropy_decisions=[])
        state['events'].append(e)
        limit=min(8,settings.max_tokens-len(state['response_token_ids']),
                  backend.max_model_len-len(prompt)-len(state['response_token_ids'])-1)
        assert limit>0
        state['pending_bridge']=dict(seed=seed,limit=limit);save(state)

def finalize_timeout(state):
    if state.get('pending_bridge'):
        e=state['events'][-1];e['bridge_stop_reason']='trace_timeout'
        if not e['bridge_token_ids']:
            del e['bridge_token_ids']
        state.pop('pending_bridge')
    state.update(status='COMPLETE',finish_reason='trace_timeout',timeout_count_as_wrong=True)
    return state

def verify_trace(x):
    assert x['status']=='COMPLETE' and x['judge_calls']==0
    ids=x['response_token_ids'];sources=x['token_sources'];assert len(ids)==len(sources)
    total=0
    for e in x['events']:
        if 'bridge_token_ids' not in e:continue
        b=e['bridge_token_ids'];pos=e['anchor_response_position'];assert 1<=len(b)<=8
        assert b[0]!=e['student_entry_token'] and ids[pos:pos+len(b)]==b
        assert sources[pos:pos+len(b)]==['teacher']*len(b)
        assert e['selected_candidate']['signed_logprob_gap_student_minus_teacher']>2
        decisions=e['entropy_decisions']
        assert [d['after_teacher_tokens'] for d in decisions]==list(range(1,len(decisions)+1))
        for d in decisions:
            assert d['continue_teacher']==(d['student_entropy']>=.5 and d['teacher_entropy']<=.2)
        assert all(d['continue_teacher'] for d in decisions[:len(b)-1])
        assert len(decisions)>=len(b)-1
        if e['bridge_stop_reason']=='entropy_condition_false':assert len(decisions)==len(b) and not decisions[-1]['continue_teacher']
        total+=len(b)
    assert sources.count('teacher')==total
