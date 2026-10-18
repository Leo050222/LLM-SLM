"""Pre-sampled, non-overlapping random handoffs; no correctness-based selection."""
from dataclasses import asdict
import random
from online_rollout.core import digest


def sample_positions(length, count, bridge_tokens, seed):
    count = min(count, length // bridge_tokens)
    rng = random.Random(seed)
    return [p + i * (bridge_tokens - 1) for i, p in enumerate(
        sorted(rng.sample(range(length - count * (bridge_tokens - 1)), count)))]


def random_rollout(row, settings, backend, checkpoint=None, save=lambda state: None):
    positions = row['random_positions']
    assert len(positions) <= 8 and all(b-a >= settings.teacher_tokens for a,b in zip(positions,positions[1:]))
    identity = digest(dict(row=row, settings=asdict(settings), policy='uniform_baseline_length_v1'))
    state = checkpoint or dict(identity=identity, response_token_ids=[], token_sources=[], events=[],
                              rounds=0, student_draft_tokens=0, discarded_tokens=0,
                              judge_calls=0, status='RUNNING')
    if state['identity'] != identity: raise ValueError('Checkpoint mismatch')
    if state['status'] == 'COMPLETE': return state
    while len(state['response_token_ids']) < settings.max_tokens:
        response = state['response_token_ids']; offset = len(response)
        prefix = row['prompt_token_ids'] + response
        capacity = min(settings.max_tokens-offset, backend.max_model_len-len(prefix)-1)
        if capacity <= 0:
            state['finish_reason']='context_limit'; break
        seed=int(digest([settings.seed,row['dataset_index'],state['rounds']])[:8],16) % (2**31)
        draft=backend.generate('student',prefix,min(settings.chunk_tokens,capacity),seed)
        ids=draft['token_ids']
        if not ids or len(ids)>min(settings.chunk_tokens,capacity):raise ValueError('Bad student draft')
        next_pos=next((p for p in positions if p>=offset),None)
        event=dict(round=state['rounds'],response_offset=offset,draft_tokens=len(ids),judgments=[])
        if next_pos is not None and next_pos < offset+len(ids):
            pos=next_pos-offset; accepted=ids[:pos]
            n=min(settings.teacher_tokens,capacity-pos)
            bridge=backend.generate('teacher',prefix+accepted,n,seed)
            bt=bridge['token_ids']; stop=bridge['finish_reason']=='stop'
            if not bt or len(bt)>n or (len(bt)!=n and not stop):raise ValueError('Bad bridge')
            event.update(anchor_response_position=next_pos,bridge_token_ids=bt,
                         requested_teacher_tokens=n,discarded_tokens=len(ids)-pos)
        else:
            accepted=ids;bt=[];stop=draft['finish_reason']=='stop'
        state['response_token_ids']=response+accepted+bt
        state['token_sources']+=['student']*len(accepted)+['teacher']*len(bt)
        state['student_draft_tokens']+=len(ids);state['discarded_tokens']+=len(ids)-len(accepted)
        state['events'].append(event);state['rounds']+=1
        if stop:state.update(status='COMPLETE',finish_reason='stop')
        save(state)
        if stop:return state
    state['status']='COMPLETE';state.setdefault('finish_reason','length');save(state);return state
