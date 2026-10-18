"""Weak-main / strong-scoring rollout state machine with prompt-repair intervention."""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import time
import uuid


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
    teacher_tokens: int = 1
    gap_threshold: float = 2.0
    max_tokens: int = 40264
    seed: int = 20260911
    gap_rule: str = "intervention_minus_main_gt_positive_threshold"
    mode: str = "weak_reprompt"
    max_takeovers: int | None = None
    first_token_ban: bool = True
    random_rates: tuple = ()

    def __post_init__(self):
        if min(self.chunk_tokens, self.teacher_tokens, self.max_tokens) <= 0:
            raise ValueError('Token limits must be positive')
        if not 0 <= self.gap_threshold < float('inf'):
            raise ValueError('Invalid gap threshold')


def _token_record(tokenizer, token_id, logprob=None):
    result = {'token_id': int(token_id),
              'token_text': tokenizer.decode([int(token_id)], skip_special_tokens=False)}
    if logprob is not None:
        result['logprob'] = float(logprob)
    return result


def rollout(row, settings, backend, judge=None, checkpoint=None, save=lambda state: None):
    """Commit only accepted prefixes; each selected anchor gets a fresh weak-main request."""
    if judge is not None:
        raise AssertionError('Judge is not part of weak-main reprompt protocol')
    identity = digest({'row': row, 'settings': asdict(settings)})
    state = checkpoint or dict(identity=identity, response_token_ids=[], token_sources=[],
                               events=[], rounds=0, main_draft_tokens=0,
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
            raise ValueError('Invalid main draft length')
        capped = settings.max_takeovers is not None and sum('bridge_token_ids' in e for e in state['events']) >= settings.max_takeovers
        candidates = [] if capped else backend.candidates(prefix, ids, settings.gap_threshold)
        if candidates:
            selected = candidates[0]
            position = selected['response_position']
            if selected['main_token_id'] != ids[position]:
                raise ValueError('Candidate token mismatch')
            accepted = ids[:position]
            offset = len(response)
            request_prefix = backend.build_reprompt_prefix(row, response, ids, position)
            if len(request_prefix) + 1 > backend.max_model_len:
                event = dict(round=state['rounds'], response_offset=offset, draft_tokens=len(ids),
                             skip_reason='reprompt_context_limit', reprompt_input_tokens=len(request_prefix),
                             draft_token_ids=list(ids), anchor_response_position=offset + position,
                             selected_candidate=selected)
                state['events'].append(event)
                state['finish_reason'] = 'reprompt_context_limit'
                break
            reprompt_seed = int(digest([seed, 'weak_reprompt'])[:8], 16) % (2**31)
            bridge_result = backend.generate('student', request_prefix, 1, reprompt_seed,
                                             banned_first=ids[position])
            bridge_ids = bridge_result['token_ids']
            if not bridge_ids or len(bridge_ids) != 1 or bridge_ids[0] == ids[position]:
                raise ValueError('Invalid weak-main replacement token')
            event = dict(round=state['rounds'], response_offset=offset, draft_tokens=len(ids),
                         draft_token_ids=list(ids), anchor_response_position=offset + position,
                         selected_candidate=selected,
                         main_sampled_token=_token_record(backend.tokenizer, ids[position], selected['main_logprob']),
                         replacement_sampled_token=_token_record(backend.tokenizer, bridge_ids[0],
                                                                  (bridge_result.get('sampled_logprobs') or [None])[0]),
                         main_top100=selected['main_top100'], intervention_top100=selected['intervention_top100'],
                         main_logprob=selected['main_logprob'],
                         intervention_logprob_on_main_token=selected['intervention_logprob_on_main_token'],
                         signed_logprob_gap_intervention_minus_main=selected['signed_logprob_gap_intervention_minus_main'],
                         reprompt_seed=reprompt_seed, reprompt_input_tokens=len(request_prefix),
                         reprompt_prompt_sha256=digest(request_prefix),
                         discarded_tokens=len(ids) - position,
                         first_token_ban=True, replacement_role='weak_main')
            accepted_stop = False
            bridge = bridge_ids
            bridge_stop = bridge_result.get('finish_reason') == 'stop'
        else:
            selected = None
            accepted, bridge = ids, []
            bridge_stop = False
            offset = len(response)
            event = dict(round=state['rounds'], response_offset=offset, draft_tokens=len(ids),
                         draft_token_ids=list(ids))
            accepted_stop = draft['finish_reason'] == 'stop'
        state['response_token_ids'] = response + accepted + bridge
        state['token_sources'] += ['weak_main'] * len(accepted) + ['weak_main_reprompt'] * len(bridge)
        state['main_draft_tokens'] += len(ids)
        state['discarded_tokens'] += len(ids) - len(accepted)
        state['judge_calls'] = 0
        state['events'].append(event)
        state['rounds'] += 1
        if selected is not None and bridge_stop:
            state['finish_reason'] = 'stop'; state['status'] = 'COMPLETE'
        elif selected is None and accepted_stop:
            state['finish_reason'] = 'stop'; state['status'] = 'COMPLETE'
        save(state)
        if state['status'] == 'COMPLETE':
            return state
    state['status'] = 'COMPLETE'
    state.setdefault('finish_reason', 'length')
    save(state)
    return state


class FileRPC:
    """Durable content-addressed requests on the shared filesystem."""
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
            time.sleep(.25)
        value = json.loads(result.read_text())
        if 'error' in value:
            raise RuntimeError(f'{role} worker failed: {value["error"]}')
        return value
