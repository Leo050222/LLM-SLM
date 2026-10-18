"""GPU-independent state machine; callbacks receive exact token-ID prefixes."""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import time
import uuid
import random


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


@dataclass(frozen=True)
class Settings:
    chunk_tokens: int = 64
    teacher_tokens: int = 8
    gap_threshold: float = 2.0
    max_tokens: int = 40264
    seed: int = 20260911
    gap_rule: str = "student_minus_teacher_gt_positive_threshold"
    mode: str = "luna_cap8"
    max_takeovers: int | None = 8
    first_token_ban: bool = True
    random_rates: tuple = ()

    def __post_init__(self):
        if min(self.chunk_tokens, self.teacher_tokens, self.max_tokens) <= 0:
            raise ValueError('Token limits must be positive')
        if not 0 <= self.gap_threshold < float('inf'):
            raise ValueError('Invalid gap threshold')




class FileRPC:
    """Durable content-addressed requests on the cluster shared filesystem.

    One worker per role/queue. Cache preserves completed requests across retries.
    """
    def __init__(self, root, timeout=1800):
        self.root, self.timeout = Path(root), timeout

    def call(self, role, payload):
        key = digest(payload)
        base = self.root / role
        request = base / 'requests' / (key + '.json')
        result = base / 'results' / (key + '.json')
        if not result.exists():
            atomic_json(request, payload)
        start = time.monotonic()
        while not result.exists():
            if time.monotonic() - start > self.timeout:
                raise TimeoutError(f'{role} RPC timed out: {key}')
            time.sleep(0.25)
        value = json.loads(result.read_text())
        if 'error' in value:
            raise RuntimeError(f'{role} worker failed: {value["error"]}')
        return value

SHIFT_SEED = 2026091802
SHIFT_VALUES = (-5,-4,-3,-2,-1,1,2,3,4,5)
SHIFT_PROTOCOL = dict(offsets=list(SHIFT_VALUES),seed=SHIFT_SEED,distribution='uniform_requested_offset',left_boundary='clip_after_last_teacher_or_response_start',right_boundary='extend_student_draft_until_target_or_normal_eos_or_length',trigger='first_signed_student_minus_teacher_gap_strictly_gt_2',ban='student_token_at_shifted_entry',zero_requested_offset=False)

def draw_shift(row, settings, round_index):
    rng=random.Random(digest([SHIFT_SEED,settings.seed,row['dataset'],row['problem_id'],row.get('original_dataset_index',row['dataset_index']),round_index]))
    return rng.choice(SHIFT_VALUES)

def rollout(row, settings, backend, judge=None, checkpoint=None, save=lambda state:None):
    assert settings.mode=='shift5' and settings.teacher_tokens==1 and settings.max_takeovers is None
    identity=digest(dict(row=row,settings=asdict(settings)))
    state=checkpoint if checkpoint is not None else dict(identity=identity,response_token_ids=[],token_sources=[],events=[],rounds=0,student_draft_tokens=0,discarded_tokens=0,judge_calls=0,status='RUNNING')
    assert state['identity']==identity
    if state['status']=='COMPLETE':return state
    prompt=row['prompt_token_ids'];limit=min(settings.max_tokens,backend.max_model_len-len(prompt)-1)
    while len(state['response_token_ids'])<limit:
        response=state['response_token_ids'];offset=len(response);prefix=prompt+response
        seed=int(digest([settings.seed,row.get('original_dataset_index',row['dataset_index']),state['rounds']])[:8],16)%(2**31)
        n=min(settings.chunk_tokens,limit-offset)
        draft=backend.generate('student',prefix,n,seed);ids=list(draft['token_ids'])
        assert 0<len(ids)<=n
        candidates=[c for c in backend.candidates(prefix,ids,settings.gap_threshold) if c['signed_logprob_gap_student_minus_teacher']>settings.gap_threshold]
        positions=[c['response_position'] for c in candidates]
        assert positions==sorted(set(positions)) and all(0<=v<len(ids) for v in positions)
        selected=candidates[0] if candidates else None
        event=dict(round=state['rounds'],response_offset=offset,draft_tokens=len(ids),judgments=[])
        full=response+ids;sources=state['token_sources']+['student']*len(ids);stop=draft['finish_reason']=='stop';finish='stop' if stop else None;discarded=0
        if selected:
            pos=selected['response_position'];assert selected['student_token_id']==ids[pos]
            trigger=offset+pos;shift=draw_shift(row,settings,state['rounds']);requested_target=trigger+shift
            # Never erase an already issued teacher intervention or prompt tokens.
            floor=max((i+1 for i,s in enumerate(state['token_sources']) if s=='teacher'),default=0)
            target=max(floor,requested_target)
            event.update(selected_candidate=selected,trigger_response_position=trigger,requested_shift=shift,
                         requested_target_position=requested_target,target_position=target,left_boundary_floor=floor,
                         left_boundary_clipped=target!=requested_target)
            # A right shift can cross the 64-token chunk: extend only to the new entry token.
            if target>=len(full) and not stop and len(full)<limit:
                extra_n=min(target-len(full)+1,limit-len(full))
                extension=backend.generate('student',prompt+full,extra_n,(seed+17)%(2**31))
                extra=list(extension['token_ids']);assert 0<len(extra)<=extra_n
                full+=extra;sources+=['student']*len(extra);ids+=extra
                event['right_extension_tokens']=len(extra)
                stop=extension['finish_reason']=='stop';finish='stop' if stop else None
            if target>=len(full):
                event['skip_reason']='eos_before_target' if stop else 'length_before_target'
            else:
                assert sources[target]=='student'
                entry=full[target]
                teacher=backend.generate('teacher',prompt+full[:target],1,seed,banned_first=entry if settings.first_token_ban else None)
                bridge=list(teacher['token_ids']);assert len(bridge)==1 and (not settings.first_token_ban or bridge[0]!=entry)
                discarded=len(full)-target
                event.update(anchor_response_position=target,actual_shift=target-trigger,bridge_token_ids=bridge,
                             requested_teacher_tokens=1,student_entry_token=entry,first_token_ban=settings.first_token_ban,
                             rollback_committed_tokens=max(0,offset-target),discarded_tokens=discarded)
                full=full[:target]+bridge;sources=sources[:target]+['teacher']
                stop=teacher['finish_reason']=='stop';finish='stop' if stop else None
        state['response_token_ids']=full;state['token_sources']=sources
        state['student_draft_tokens']+=len(ids);state['discarded_tokens']+=discarded
        state['events'].append(event);state['rounds']+=1
        if stop:state.update(status='COMPLETE',finish_reason=finish)
        save(state)
        if stop:return state
    state.update(status='COMPLETE',finish_reason='length' if len(state['response_token_ids'])>=settings.max_tokens else 'context_limit')
    save(state);return state
