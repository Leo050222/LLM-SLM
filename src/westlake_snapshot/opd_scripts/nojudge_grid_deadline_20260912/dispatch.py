import os, sys, json, time, queue, threading, fcntl, hashlib, copy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from dataclasses import asdict
from deadline import Budget, TraceDeadline, consume_tasks

ROOT = Path('/storage/yukaichengLab/wangxin/test_eiloeo')
S = ROOT / 'opd_scripts/nojudge_grid_20260912'
sys.path.insert(0, str(S))
from ablation_core import Settings, atomic_json, digest, rollout
from run import Backend, tokenizer_fingerprint


class DeadlineRPC:
    def __init__(self, root, budget):
        self.root, self.budget = root, budget

    def call(self, role, payload):
        self.budget.check()
        key = digest(payload)
        base = self.root / role
        request, result = base / 'requests' / (key + '.json'), base / 'results' / (key + '.json')
        if not result.exists():
            atomic_json(request, payload)
        try:
            while not result.exists():
                self.budget.check()
                time.sleep(.1)
            self.budget.check()
        except TraceDeadline:
            # Stop future work for this trace; an already executing GPU batch may finish.
            request.unlink(missing_ok=True)
            raise
        value = json.loads(result.read_text())
        if 'error' in value:
            raise RuntimeError(value['error'])
        return value


def main():
    from transformers import AutoTokenizer
    P = Path(os.environ['GRID_ROOT'])
    slot = Path(os.environ['OPD_ONLINE_RUN'])
    Q = slot / 'queue'
    lock = (P / 'deadline_scheduler.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for _ in range(720):
        if len(list(Q.glob('replica_*/*/model.json'))) == 4:
            break
        time.sleep(5)
    else:
        raise RuntimeError('Models not ready')
    models = [{role: json.loads((Q / f'replica_{i}' / role / 'model.json').read_text())
               for role in ['student', 'teacher']} for i in range(2)]
    assert models[0] == models[1]
    tokenizer = AutoTokenizer.from_pretrained(models[0]['student']['model'], local_files_only=True)
    assert all(m['tokenizer_sha256'] == tokenizer_fingerprint(tokenizer) for m in models[0].values())
    rows = [json.loads(l) for l in (P / 'prompts.jsonl').read_text().splitlines()]
    assert len(rows) == 2254
    configs = json.loads((P / 'configs.json').read_text())
    baseline = json.loads((P / 'k1_t2/manifest.json').read_text())
    assert baseline['models'] == models[0]
    tasks = queue.Queue()
    remaining, settings_by_group, counts = {}, {}, Counter()
    runtime = dict(trace_limit_seconds=900, timeout_policy='preserve_partial_count_wrong_no_retry',
                   scheduler='shared task queue across configurations; two replica pools of32',
                   replicas=2, concurrency=64,
                   code_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
    for c in configs:
        g = c['name']; d = P / g
        if (d / 'VERIFIED.json').exists():
            continue
        m = copy.deepcopy(baseline)
        m['settings']['teacher_tokens'] = c['k']; m['settings']['gap_threshold'] = float(c['threshold'])
        m['execution'] = dict(replicas=2, concurrency=64)
        f = d / 'manifest.json'
        if f.exists():
            old = json.loads(f.read_text())
            for key in ['settings','input_sha256','models','judge_config','sampling','expected_rows','thinking','first_token_ban']:
                assert old[key] == m[key], (g, key)
            m = old
            if 'runtime_policy' not in m:
                atomic_json(d / 'manifest_before_deadline.json', m)
        m['runtime_policy'] = runtime
        atomic_json(f, m)
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
            if x.get('finish_reason') == 'trace_timeout':
                assert x['timeout_count_as_wrong'] and x['trace_elapsed_seconds'] >= 900
                count += 1
        atomic_json(d / 'VERIFIED.json', dict(rows=2254, verified_at=time.time(), timeout_rows=count, runtime_policy=runtime))
        print('VERIFIED', g, 'timeout', count, flush=True)

    def action(task, replica):
        g, row = task; d = P / g; i = row['dataset_index']; path = d / 'traces' / f'{i:05d}.json'
        settings = settings_by_group[g]
        state = json.loads(path.read_text()) if path.exists() else dict(identity=digest(dict(row=row, settings=asdict(settings))),
                 response_token_ids=[], token_sources=[], events=[], rounds=0, student_draft_tokens=0,
                 discarded_tokens=0, judge_calls=0, status='RUNNING')
        budget = Budget(d / 'timings' / path.name)
        backend = Backend(DeadlineRPC(Q / f'replica_{replica}', budget), models[replica], tokenizer)
        print('TRACE_START', g, i, 'replica', replica, flush=True)
        def save(value):
            budget.persist()
            atomic_json(path, value)
        try:
            try:
                if state['status'] != 'COMPLETE':
                    budget.check()
                result = rollout(row, settings, backend, None, state, save)
            except TraceDeadline:
                # Only committed tokens are kept; no new rounds and no retry.
                result = state
                result.update(status='COMPLETE', finish_reason='trace_timeout', timeout_count_as_wrong=True)
            budget.persist()
            result['trace_elapsed_seconds'] = budget.elapsed()
            result['trace_timing_scope'] = 'since deadline policy activation; cumulative active wall time'
            save(result)
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
            print('TRACE_ERROR', g, i, type(e).__name__, flush=True)

    for g, n in remaining.items():
        if n == 0:
            verify(g)
    with ThreadPoolExecutor(max_workers=64) as pool:
        fs = [pool.submit(consume_tasks, tasks, action, i % 2) for i in range(64)]
        for f in fs:
            f.result()
    atomic_json(slot / 'run_status.json', dict(failures=failures, counts=dict(counts)))
    if failures:
        atomic_json(slot / 'FAILED', dict(failures=failures))
        raise RuntimeError('Inspect trace failures; no automatic repeated retry')
    assert all((P / c['name'] / 'VERIFIED.json').exists() for c in configs)
    atomic_json(P / 'ALL_GENERATION_COMPLETE.json', dict(rows=56350, configs=25, completed_at=time.time(), runtime_policy=runtime))
    atomic_json(slot / 'ALL_GENERATION_COMPLETE.json', dict(slot_done=True))

if __name__ == '__main__':
    main()
