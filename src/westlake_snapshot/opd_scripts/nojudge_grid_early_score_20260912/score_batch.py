import os, sys, json, hashlib, importlib.util, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict, Counter

ROOT = Path('/storage/yukaichengLab/wangxin/test_eiloeo')
P = ROOT / 'opd_eval_hard/nojudge_grid_20260912'
SOURCE = ROOT / 'opd_scripts/nojudge_grid_20260912/score.py'
spec = importlib.util.spec_from_file_location('original_grid_score', SOURCE)
scorer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scorer
spec.loader.exec_module(scorer)
scorer.O = P

def score_one(arg):
    g, i = arg
    f = P / g / 'completed' / f'{i:05d}.json'
    raw = f.read_bytes()
    r = json.loads(raw)
    if r.get('finish_reason') != 'trace_timeout':
        return scorer.one(arg)
    assert r['timeout_count_as_wrong'] and r['trace_elapsed_seconds'] >= 900
    s = dict(group=g, dataset=r['dataset'], subset=r['subset'], dataset_index=i,
             problem_id=r['problem_id'], correct=False, predicted=None, status='trace_timeout',
             source_sha256=hashlib.sha256(raw).hexdigest(), tokens=len(r['response_token_ids']))
    dest = P / 'scores' / g / f.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        assert json.loads(dest.read_text()) == s
    else:
        tmp = dest.with_suffix('.tmp'); tmp.write_text(json.dumps(s)); tmp.replace(dest)
    return s

def main():
    batch = Path(sys.argv[1])
    manifest = json.loads((batch / 'batch.json').read_text())
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == manifest['scorer_sha256']
    rows = [(g, i) for g in manifest['groups'] for i in manifest['indices']]
    for g in manifest['groups']:
        assert (P / g / 'VERIFIED.json').exists()
        assert {int(f.stem) for f in (P / g / 'completed').glob('*.json')} == set(manifest['indices'])
    if len(sys.argv) > 2 and sys.argv[2] == 'summarize':
        for rank in range(3):
            assert (batch / f'done_{rank}.json').exists()
        groups = defaultdict(list)
        for g, i in rows:
            f = P / g / 'completed' / f'{i:05d}.json'
            s = json.loads((P / 'scores' / g / f.name).read_text())
            assert s['source_sha256'] == hashlib.sha256(f.read_bytes()).hexdigest()
            assert s['group'] == g and s['dataset_index'] == i
            groups[(g, s['dataset'])].append(s)
        summary = {}
        for (g, ds), rs in groups.items():
            expected = {'math500': 500, 'olympiad_bench': 674, 'bbh': 1080}[ds]
            assert len(rs) == expected
            correct = sum(r['correct'] for r in rs)
            summary.setdefault(g, {})[ds] = dict(questions=expected, correct=correct, accuracy=correct/expected, score_status=dict(Counter(r['status'] for r in rs)))
        (batch / 'summary.json').write_text(json.dumps(summary, indent=2))
        (batch / 'SCORING_COMPLETE.json').write_text(json.dumps(dict(rows=len(rows), groups=manifest['groups'], completed_at=time.time())))
        print(json.dumps(summary), flush=True)
        return
    rank = int(os.environ['SLURM_PROCID'])
    assigned = rows[rank::3]
    with ProcessPoolExecutor(max_workers=60) as pool:
        fs = [pool.submit(score_one, row) for row in assigned]
        for n, f in enumerate(as_completed(fs), 1):
            f.result()
            if n % 100 == 0:
                print('SCORED', rank, n, len(assigned), flush=True)
    (batch / f'done_{rank}.json').write_text(json.dumps(dict(rows=len(assigned), completed_at=time.time())))

if __name__ == '__main__':
    main()
