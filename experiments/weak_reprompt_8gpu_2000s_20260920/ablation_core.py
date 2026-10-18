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


def rollout(row, settings, backend, judge=None, checkpoint=None, save=lambda state: None):
    """Commit only accepted prefixes. Never reuse candidates across interventions."""
    identity = digest({'row': row, 'settings': asdict(settings)})
    state = checkpoint or dict(identity=identity, response_token_ids=[], token_sources=[],
                               events=[], rounds=0, student_draft_tokens=0,
                               discarded_tokens=0, judge_calls=0, status='RUNNING')
    if state['identity'] != identity:
        raise ValueError('Checkpoint input/config mismatch')
    if state['status'] == 'COMPLETE':
        return state
    prompt = row['prompt_token_ids']
    while len(state['response_token_ids']) < settings.max_tokens:
        response = state['response_token_ids']
        prefix = prompt + response
        capacity = min(settings.max_tokens - len(response), backend.max_model_len - len(prefix) - 1)
        if capacity <= 0:
            state['finish_reason'] = 'context_limit'
            break
        seed = int(digest([settings.seed, row.get('original_dataset_index', row['dataset_index']), state['rounds']])[:8], 16) % (2**31)
        draft = backend.generate('student', prefix, min(settings.chunk_tokens, capacity), seed)
        ids = draft['token_ids']
        if not ids or len(ids) > min(settings.chunk_tokens, capacity):
            raise ValueError('Invalid student draft length')
        capped = settings.max_takeovers is not None and sum('bridge_token_ids' in e for e in state['events']) >= settings.max_takeovers
        if capped:
            candidates = []
        elif settings.mode == 'random_allow':
            probability = dict(settings.random_rates)[row['dataset']]
            rng = random.Random(digest([settings.seed, row.get('original_dataset_index', row['dataset_index']), state['rounds'], 'random_anchor']))
            candidates = []
            for pos, token in enumerate(ids):
                if rng.random() < probability:
                    candidates = [dict(response_position=pos, student_token_id=token, random_trigger_probability=probability)]
                    break
        else:
            candidates = [c for c in backend.candidates(prefix, ids, settings.gap_threshold)
                          if c['signed_logprob_gap_student_minus_teacher'] > settings.gap_threshold]
        positions = [c['response_position'] for c in candidates]
        if positions != sorted(set(positions)) or any(p < 0 or p >= len(ids) for p in positions):
            raise ValueError('Invalid candidate order/position')
        selected = None
        judgments = []
        for candidate in candidates:
            position = candidate['response_position']
            if candidate['student_token_id'] != ids[position]:
                raise ValueError('Candidate token mismatch')
            if settings.mode in ('luna_cap8', 'luna_allow'):
                answer = judge(row, prefix + ids[:position], candidate)
                judgments.append({'candidate': candidate, 'judgment': answer})
            else:
                answer = {'select_anchor': True}
            if answer['select_anchor']:
                selected = candidate
                break
        offset = len(response)
        event = dict(round=state['rounds'], response_offset=offset, draft_tokens=len(ids),
                     judgments=judgments)
        if selected is None:
            accepted, bridge = ids, []
            stop = draft['finish_reason'] == 'stop'
        else:
            position = selected['response_position']
            accepted = ids[:position]
            handoff_prefix = prefix + accepted
            # Hard budget/context boundary is the only shortening of k; EOS may stop early.
            k = min(settings.teacher_tokens, settings.max_tokens - offset - position,
                    backend.max_model_len - len(handoff_prefix) - 1)
            from reprompt import build_prompt, TEMPLATE_SHA256
            request_prefix = build_prompt(backend.tokenizer, row, response, ids, position)
            if len(request_prefix) + 1 > backend.max_model_len:
                event.update(skip_reason='reprompt_context_limit', reprompt_input_tokens=len(request_prefix),
                             draft_token_ids=ids, anchor_response_position=offset+position)
                state['events'].append(event)
                state['finish_reason']='reprompt_context_limit'
                break
            reprompt_seed = int(digest([seed, 'weak_reprompt'])[:8],16) % (2**31)
            teacher = backend.call('student', op='generate', prefix=request_prefix, n=1,
                                   seed=reprompt_seed,
                                   banned_first=ids[position] if settings.first_token_ban else None)
            event.update(draft_token_ids=list(ids), reprompt_seed=reprompt_seed,
                         reprompt_input_tokens=len(request_prefix), reprompt_prompt_sha256=digest(request_prefix),
                         template_sha256=TEMPLATE_SHA256, reprompt_generation=teacher)
            bridge = teacher['token_ids']
            if not bridge or len(bridge) > k or (settings.first_token_ban and bridge[0] == ids[position]):
                raise ValueError('Invalid teacher bridge')
            if len(bridge) != k and teacher['finish_reason'] != 'stop':
                raise ValueError('Teacher bridge ended without EOS before k')
            stop = teacher['finish_reason'] == 'stop'
            event.update(anchor_response_position=offset + position,
                         bridge_token_ids=bridge, requested_teacher_tokens=k,
                         selected_candidate=selected, student_entry_token=ids[position],
                         first_token_ban=settings.first_token_ban,
                         discarded_tokens=len(ids) - position)
        state['response_token_ids'] = response + accepted + bridge
        state['token_sources'] += ['student'] * len(accepted) + ['student_reprompt'] * len(bridge)
        state['student_draft_tokens'] += len(ids)
        state['discarded_tokens'] += len(ids) - len(accepted)
        state['judge_calls'] += len(judgments)
        state['events'].append(event)
        state['rounds'] += 1
        if stop:
            state['finish_reason'] = 'stop'
            state['status'] = 'COMPLETE'
        save(state)
        if stop:
            return state
    state['status'] = 'COMPLETE'
    state.setdefault('finish_reason', 'length')
    save(state)
    return state


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

from reprompt import REPROMPT_PROTOCOL
