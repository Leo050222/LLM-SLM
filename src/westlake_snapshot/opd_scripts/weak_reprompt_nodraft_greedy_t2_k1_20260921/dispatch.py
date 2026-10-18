import os, sys, json, time, queue, threading, fcntl, hashlib, copy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from dataclasses import asdict
from deadline import Budget, TraceDeadline, consume_tasks

ROOT = Path('/storage/yukaichengLab/wangxin/test_eiloeo')
S = Path(__file__).parent
sys.path.insert(0, str(S))
from ablation_core import Settings, atomic_json, digest, rollout, REPROMPT_PROTOCOL
from run import Backend, tokenizer_fingerprint


from queue_rpc import RPC as DeadlineRPC
from migration import migrate


def baseline_rollout(row, settings, backend, state, save, role):
    assert state['identity'] == digest(dict(row=row,settings=asdict(settings)))
    if state['status']=='COMPLETE': return state
    while len(state['response_token_ids']) < settings.max_tokens:
        prefix=row['prompt_token_ids']+state['response_token_ids']
        n=min(settings.chunk_tokens,settings.max_tokens-len(state['response_token_ids']),backend.max_model_len-len(prefix)-1)
        if n<=0:
            state['finish_reason']='context_limit'; break
        seed=int(digest([settings.seed,row['dataset_index'],state['rounds']])[:8],16)%(2**31)
        out=backend.call(role,op='generate',prefix=prefix,n=n,seed=seed)
        ids=out['token_ids']
        assert 0<len(ids)<=n
        state['response_token_ids']+=ids
        state['token_sources'] += [role]*len(ids)
        state['rounds']+=1
        if role=='student': state['student_draft_tokens']+=len(ids)
        save(state)
        if out['finish_reason']=='stop':
            state['finish_reason']='stop'; break
    else: state['finish_reason']='length'
    state['status']='COMPLETE'; save(state)
    return state


def main():
    from transformers import AutoTokenizer
    P = Path(os.environ['GRID_ROOT'])
    slot = Path(os.environ['OPD_ONLINE_RUN'])
    Q = slot / 'queue'
    replicas=int(os.environ.get('GRID_REPLICAS','2'));concurrency=32*replicas
    assert replicas == 2
    lock = (P / 'deadline_scheduler.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for _ in range(720):
        if len(list(Q.glob('replica_*/*/model.json'))) == replicas*2:
            break
        time.sleep(5)
    else:
        raise RuntimeError('Models not ready')
    models = [{role: json.loads((Q / f'replica_{i}' / role / 'model.json').read_text())
               for role in ['student', 'teacher']} for i in range(replicas)]
    assert all(m == models[0] for m in models)
    tokenizer = AutoTokenizer.from_pretrained(models[0]['student']['model'], local_files_only=True)
    assert all(m['tokenizer_sha256'] == tokenizer_fingerprint(tokenizer) for m in models[0].values())
    rows = [json.loads(l) for l in (P / 'prompts.jsonl').read_text().splitlines()]
    assert len(rows) == int(os.environ['EXPECTED_ROWS'])
    configs = json.loads((P / 'configs.json').read_text())
    baseline = dict(models=models[0], settings=asdict(Settings(mode='nojudge_grid', max_takeovers=None)), input_sha256=digest(rows), judge_config=None, sampling=dict(temperature=0.,top_p=1.,decoding='greedy',replacement='argmax_after_original_token_ban',score_temperature=1.),expected_rows=len(rows),thinking=False,first_token_ban=True)
    if (P / 'base_settings.json').exists():
        baseline['settings'] = json.loads((P / 'base_settings.json').read_text())
    tasks = queue.Queue()
    remaining, settings_by_group, counts = {}, {}, Counter()
    config_by_group = {c['name']:c for c in configs}
    runtime = dict(trace_limit_seconds=2000, timeout_policy='preserve_partial_count_wrong_no_retry',
                   scheduler='FIFO oldest-first homogeneous RPC batches; per-waiter leases;32 traces per replica',
                   replicas=replicas, concurrency=concurrency,
                   code_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
    for c in configs:
        g = c['name']; d = P / g
        if (d / 'VERIFIED.json').exists():
            continue
        m = copy.deepcopy(baseline)
        m['settings']['teacher_tokens'] = c.get('k', 1); m['settings']['gap_threshold'] = float(c.get('threshold', 2))
        m['settings']['mode'] = c.get('role', 'weak_reprompt')
        m['settings']['seed'] = c['seed']
        m['replicate_index'] = c['replicate_index']
        m['base_config'] = c['base_config']
        m['baseline_generation'] = 'unconstrained 64-token chunks; same per-round seed as intervention runs' if 'role' in c else None
        m['execution'] = dict(replicas=replicas, concurrency=concurrency)
        m = json.loads(json.dumps(m))  # Compare canonical JSON types on resume.
        f = d / 'manifest.json'
        m['runtime_policy'] = runtime
        m['reprompt_protocol'] = REPROMPT_PROTOCOL
        migrate(f, m)
        settings = dict(m['settings']); settings['random_rates'] = tuple(tuple(a) for a in settings['random_rates'])
        settings_by_group[g] = Settings(**settings)
        pending = [row for row in rows if not (d / 'completed' / f"{row['dataset_index']:05d}.json").exists()]
        remaining[g] = len(pending)
        for row in pending:
            tasks.put((g, row))
        print('QUEUED', g, len(pending), flush=True)
    atomic_json(slot / 'runtime_policy.json', runtime)
    state_lock = threading.Lock()
    failures = []

    def verify(g):
        d = P / g; settings = settings_by_group[g]; count = 0
        for row in rows:
            x = json.loads((d / 'completed' / f"{row['dataset_index']:05d}.json").read_text())
            assert x['status'] == 'COMPLETE' and x['judge_calls'] == 0
            assert x['identity'] == digest(dict(row=row, settings=asdict(settings)))
            assert x['prompt_token_ids'] == row['prompt_token_ids'] and x['problem_id'] == row['problem_id']
            assert len(x['token_sources']) == len(x['response_token_ids'])
            for e in x['events']:
                assert not e['judgments']
                if 'bridge_token_ids' in e:
                    assert 0 < len(e['bridge_token_ids']) <= settings.teacher_tokens
                    assert e['bridge_token_ids'][0] != e['student_entry_token']
                    assert e['selected_candidate']['signed_logprob_gap_student_minus_teacher'] > settings.gap_threshold
                    assert e['template_sha256'] == REPROMPT_PROTOCOL['template_sha256']
                    assert e['reprompt_input_tokens'] + 1 <= models[0]['student']['max_model_len']
                    assert e['draft_token_ids'][e['selected_candidate']['response_position']] == e['student_entry_token']
            if x.get('finish_reason') == 'trace_timeout':
                assert x['timeout_count_as_wrong'] and x['trace_elapsed_seconds'] >= 2000
                count += 1
        atomic_json(d / 'VERIFIED.json', dict(rows=len(rows), verified_at=time.time(), timeout_rows=count, runtime_policy=runtime))
        print('VERIFIED', g, 'timeout', count, flush=True)

    def action(task, replica):
        g, row = task; d = P / g; i = row['dataset_index']; path = d / 'traces' / f'{i:05d}.json'
        settings = settings_by_group[g]
        state = json.loads(path.read_text()) if path.exists() else dict(identity=digest(dict(row=row, settings=asdict(settings))),
                 response_token_ids=[], token_sources=[], events=[], rounds=0, student_draft_tokens=0,
                 discarded_tokens=0, judge_calls=0, status='RUNNING')
        resumed_checkpoint = path.exists()
        budget = Budget(d / 'timings' / path.name, limit=2000)
        backend = Backend(DeadlineRPC(Q / f'replica_{replica}', budget), models[replica], tokenizer)
        print('TRACE_START', g, i, 'replica', replica, flush=True)
        def save(value):
            budget.persist()
            atomic_json(path, value)
        try:
            try:
                if state['status'] != 'COMPLETE':
                    budget.check()
                if 'role' in config_by_group[g]:
                    result = baseline_rollout(row, settings, backend, state, save, config_by_group[g]['role'])
                else:
                    result = rollout(row, settings, backend, None, state, save)
            except TraceDeadline:
                # Only committed tokens are kept; no new rounds and no retry.
                result = state
                result.update(status='COMPLETE', finish_reason='trace_timeout', timeout_count_as_wrong=True)
            budget.persist()
            result['runtime_version'] = 'fifo_v2_20260916'
            result['resumed_token_checkpoint'] = resumed_checkpoint
            result['trace_elapsed_seconds'] = budget.elapsed()
            result['trace_timing_scope'] = 'since deadline policy activation; cumulative active wall time'
            save(result)
            events=[e for e in result['events'] if 'bridge_token_ids' in e]
            assert [e['anchor_response_position'] for e in events] == [j for j,v in enumerate(result['token_sources']) if v=='student_reprompt']
            result['intervention_audit']=dict(actual_total=len(events),
                reprompt_prefill_tokens=sum(e['reprompt_input_tokens'] for e in events),
                strong_generated_tokens=0)
            output = dict(row, **result)
            output['response'] = tokenizer.decode(result['response_token_ids'], skip_special_tokens=True)
            atomic_json(d / 'completed' / path.name, output)
            with state_lock:
                remaining[g] -= 1; counts[g] += 1
                print('TRACE_DONE', g, i, result['finish_reason'], round(budget.elapsed(), 2), flush=True)
                if remaining[g] == 0:
                    verify(g)
        except Exception as e:
            budget.persist()
            with state_lock:
                failures.append(dict(group=g, index=i, error_type=type(e).__name__))
            print('TRACE_ERROR', g, i, repr(e), flush=True)

    for g, n in remaining.items():
        if n == 0:
            verify(g)
    segment_end = float('inf')
    def consume_segment(replica):
        while time.monotonic() < segment_end:
            try: task = tasks.get_nowait()
            except queue.Empty: return
            action(task, replica)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        fs = [pool.submit(consume_segment, i % replicas) for i in range(concurrency)]
        for f in fs:
            f.result()
    atomic_json(slot / 'run_status.json', dict(failures=failures, counts=dict(counts)))
    if failures:
        atomic_json(slot / 'FAILED', dict(failures=failures))
        raise RuntimeError('Inspect trace failures; no automatic repeated retry')
    if not tasks.empty():
        atomic_json(slot / 'SEGMENT_DONE.json',dict(pending=tasks.qsize()))
        return
    assert all((P / c['name'] / 'VERIFIED.json').exists() for c in configs)
    atomic_json(P / 'ALL_GENERATION_COMPLETE.json', dict(rows=len(rows)*len(configs), configs=len(configs), completed_at=time.time(), runtime_policy=runtime))
    atomic_json(slot / 'ALL_GENERATION_COMPLETE.json', dict(slot_done=True))
    atomic_json(slot / 'SEGMENT_DONE.json',dict(pending=0))

if __name__ == '__main__':
    main()
