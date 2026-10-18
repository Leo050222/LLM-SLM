"""Score only the 800 authorized CRUXEval-I continuations, locally in bwrap."""
import concurrent.futures
import collections
import fcntl
import hashlib
import json
import time
from pathlib import Path
from ablation_core import atomic_json, digest
from branch import verify
from code_scorer import sandbox, extract

R = Path('/storage/yukaichengLab/wangxin/test_eiloeo')
P = R/'opd_eval_prefix/cruxeval_i_fresh100_cap1_20260917'

def main():
    assert json.loads((P/'ALL_GENERATION_COMPLETE.json').read_text())['rows'] == 800
    lock = (P/'scoring.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    raw = (P/'prompts.jsonl').read_bytes()
    meta = json.loads((P/'INPUTS_COMPLETE.json').read_text())
    assert hashlib.sha256(raw).hexdigest() == meta['sha256']
    rows = list(map(json.loads, raw.splitlines()))
    assert len(rows) == 100 and meta['caps'] == [1]
    data = {x['id']: x for x in map(json.loads, (R/'opd_eval_code/code_reasoning_grid_20260913/data/cruxeval.jsonl').read_text().splitlines())}
    sig = digest({n: hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ['score.py', 'code_scorer.py', 'score_one.py']})

    def action(task):
        row, rep = task
        f = P/f'cap1_r{rep}'/'completed'/f"{row['dataset_index']:05d}.json"
        raw = f.read_bytes()
        source = hashlib.sha256(raw).hexdigest()
        x = json.loads(raw)
        assert x['dataset'] == 'cruxeval_i' and x['problem_id'] == row['problem_id']
        assert x['cap'] == 1 and x['replicate'] == rep and x['seed'] == meta['branch_seeds'][rep]
        assert verify(row, 1, x) == x['actual_takeovers'] == 1
        dest = f.parent.parent/'scores'/f.name
        if dest.exists():
            result = json.loads(dest.read_text())
            assert result['source_sha256'] == source and result['scoring_signature'] == sig
            assert type(result['correct']) is bool
            return result
        result = sandbox(dict(dataset='cruxeval_i', problem=data[x['problem_id']], solution=extract('cruxeval_i', x['response'])))
        assert type(result['correct']) is bool
        result.update(dataset=x['dataset'], problem_id=x['problem_id'], sample_index=row['dataset_index'], cap=1, replicate=rep, actual_takeovers=1, source_sha256=source, scoring_signature=sig, completed_at=time.time())
        atomic_json(dest, result)
        return result

    tasks = [(row, rep) for row in rows for rep in range(8)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(action, tasks))
    assert len(results) == len({(x['sample_index'], x['replicate']) for x in results}) == 800
    summary = dict(dataset='cruxeval_i', cap=1, n=800, correct=sum(x['correct'] for x in results), accuracy=sum(x['correct'] for x in results)/800, mean_actual_takeovers=1., statuses=dict(collections.Counter(x['status'] for x in results)), scoring_signature=sig)
    atomic_json(P/'summary.json', summary)
    atomic_json(P/'ALL_SCORING_COMPLETE.json', dict(rows=800, completed_at=time.time()))
    print('ALL_SCORING_COMPLETE', json.dumps(summary), flush=True)

if __name__ == '__main__':
    main()
