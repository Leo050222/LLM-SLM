import collections
import concurrent.futures
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from code_scorer import sandbox, extract
import judge_client

R = Path('/storage/yukaichengLab/wangxin/test_eiloeo')
OUT = R / 'opd_eval_ablation/strong_reprompt_score_20261005'
S = Path(__file__).parent
EXPERIMENTS = [
    dict(label='strong8_weak17', root=R / 'opd_eval_ablation/strong8_weak17_reprompt_greedy_t2_k1_20261005/r0_strong_reprompt_t2_k1'),
    dict(label='strong4_weak17', root=R / 'opd_eval_ablation/strong4_weak17_reprompt_greedy_t2_k1_20261005/r0_strong_reprompt_t2_k1'),
]
CODE_DATASETS = {'cruxeval_o', 'cruxeval_i', 'humaneval_plus', 'mbpp_plus', 'livecodebench'}
MATH_DATASETS = {'math500', 'bbh', 'olympiad_bench', 'aime24', 'aime25'}

import sys
sys.path.insert(0, str(R / 'eval_repos/evalscope'))
from evalscope.metrics.math.parser import extract_answer
from evalscope.benchmarks.bbh.bbh_adapter import BBHAdapter
from evalscope.benchmarks.olympiad_bench.olympiad_bench_adapter import OlympiadBenchAdapter


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def candidate(row):
    response = row['response'].rsplit('</think>', 1)[-1].strip()
    ds = row['dataset']
    if ds == 'math500':
        return extract_answer(response)
    if ds == 'bbh':
        fn = BBHAdapter._extract_mc_answer if row['metadata']['multiple_choice'] else BBHAdapter._extract_ff_answer
        return fn(response)
    if ds == 'olympiad_bench':
        return OlympiadBenchAdapter.extract_answer(None, response, SimpleNamespace(metadata=row['metadata']))
    if ds in {'aime24', 'aime25'}:
        # Keep the full final response so gpt-4o-mini can identify the explicitly selected answer.
        return response
    raise ValueError(ds)


def reference(row):
    if row['dataset'] == 'olympiad_bench':
        return row['metadata']['final_answer'][0]
    return row['solution']


def load_code_data():
    data = {}
    hashes = {}
    specs = [
        ('cruxeval_o', 'cruxeval.jsonl', 'id'),
        ('cruxeval_i', 'cruxeval.jsonl', 'id'),
        ('humaneval_plus', 'HumanEvalPlus.jsonl', 'task_id'),
        ('mbpp_plus', 'MbppPlus.jsonl', 'task_id'),
        ('livecodebench', 'livecodebench_v6_incremental.jsonl', 'question_id'),
    ]
    for ds, name, key in specs:
        path = R / 'opd_eval_code/code_reasoning_grid_20260913/data' / name
        hashes[name] = sha(path.read_bytes())
        items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        data[ds] = {(item['platform'] + '/' + item[key] if ds == 'livecodebench' else item[key]): item for item in items}
    return data, hashes


def main():
    if not (OUT / 'judge/EXTERNAL_SCORING_APPROVED.json').exists():
        raise RuntimeError('Missing external scoring authorization')
    if not all((exp['root'] / 'VERIFIED.json').exists() for exp in EXPERIMENTS):
        raise RuntimeError('Rollout verification is incomplete')
    prompt = (OUT / 'judge/system_prompt.txt').read_text()
    port = json.loads((OUT / 'judge/relay.json').read_text())['port']
    client = judge_client.Client(port, 64, prompt)
    code_data, data_hashes = load_code_data()
    scoring_signature = sha(json.dumps(dict(
        judge_model='gpt-4o-mini', prompt_sha256=sha(prompt.encode()), data_hashes=data_hashes,
        code_sha256={name: sha((S / name).read_bytes()) for name in ['score.py', 'code_scorer.py', 'score_one.py', 'judge_client.py']},
        timeout_policy='trace_timeout_count_wrong', math_workers=64, code_workers=24,
    ), sort_keys=True).encode())
    atomic_json(OUT / 'SCORING_SMOKE.json', dict(passed=True, scoring_signature=scoring_signature))

    files = {'code': [], 'math': []}
    expected_each = 4631
    for exp in EXPERIMENTS:
        rows = sorted((exp['root'] / 'completed').glob('*.json'))
        if len(rows) != expected_each:
            raise RuntimeError(f"{exp['label']} expected {expected_each} completed rows, found {len(rows)}")
        for path in rows:
            row = json.loads(path.read_text())
            ds = row['dataset']
            if ds in CODE_DATASETS:
                family = 'code'
            elif ds in MATH_DATASETS:
                family = 'math'
            else:
                raise ValueError(f'Unknown dataset {ds}')
            files[family].append((exp, path))
    if len(files['code']) != 4634 or len(files['math']) != 4628:
        raise RuntimeError(f"Unexpected family counts: code={len(files['code'])}, math={len(files['math'])}")

    def action(item):
        exp, path = item
        raw = path.read_bytes()
        source_sha256 = sha(raw)
        dest = path.parent.parent / 'scores' / path.name
        if dest.exists():
            old = json.loads(dest.read_text())
            if old['source_sha256'] != source_sha256 or old['scoring_signature'] != scoring_signature:
                raise RuntimeError(f'Score identity mismatch for {path}')
            if type(old.get('correct')) is bool:
                return old
        row = json.loads(raw)
        if row.get('status') != 'COMPLETE':
            raise RuntimeError(f"Incomplete rollout {path}")
        if row.get('finish_reason') == 'trace_timeout':
            result = dict(correct=False, status='trace_timeout')
        elif exp['label'] and row['dataset'] in CODE_DATASETS:
            result = sandbox(dict(dataset=row['dataset'], problem=code_data[row['dataset']][row['problem_id']], solution=extract(row['dataset'], row['response'])))
        else:
            predicted = candidate(row)
            target = reference(row)
            if predicted is None or not str(predicted).strip():
                result = dict(correct=False, status='extraction_missing', is_correct=False, reason='empty extracted answer')
            else:
                result = client.call(json.dumps(dict(candidate_answer=predicted, reference_answer=target), ensure_ascii=False))
                if result.get('status') == 'api_error':
                    raise RuntimeError('API error; leave this item resumable')
                result['correct'] = result['is_correct']
            result.update(predicted=predicted, reference=target)
        if type(result.get('correct')) is not bool:
            raise RuntimeError(f'Non-boolean score for {path}')
        result.update(
            dataset=row['dataset'], problem_id=row.get('problem_id'), dataset_index=row['dataset_index'],
            experiment=exp['label'], source_sha256=source_sha256,
            scoring_signature=scoring_signature, completed_at=time.time(),
        )
        atomic_json(dest, result)
        return result

    def group(family, workers):
        pending = list(files[family])
        for attempt in range(3):
            failed = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(action, item): item for item in pending}
                for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    try:
                        future.result()
                    except Exception:
                        failed.append(futures[future])
                    if index % 250 == 0:
                        atomic_json(OUT / 'progress.json', dict(family=family, attempt=attempt + 1, processed=index, total=len(pending), errors=len(failed), updated_at=time.time()))
                        print('SCORE_PROGRESS', family, attempt + 1, index, len(pending), 'errors', len(failed), flush=True)
            if not failed:
                return
            pending = failed
            time.sleep(30)
        atomic_json(OUT / f'{family}_SCORING_ERRORS.json', [dict(experiment=e['label'], file=str(path)) for e, path in pending])
        raise RuntimeError(f'Unresolved {family} scoring errors: {len(pending)}')

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(group, 'code', 24), pool.submit(group, 'math', 64)]
        for future in futures:
            future.result()

    counts = collections.defaultdict(collections.Counter)
    usage = collections.Counter()
    for family in ['code', 'math']:
        for exp, path in files[family]:
            score = json.loads((path.parent.parent / 'scores' / path.name).read_text())
            key = (exp['label'], score['dataset'])
            counts[key]['n'] += 1
            counts[key]['correct'] += int(score['correct'])
            if score.get('status') == 'ok':
                for k, value in score.get('usage', {}).items():
                    if isinstance(value, int):
                        usage[k] += value
    groups = [dict(experiment=exp, dataset=ds, **values, accuracy=values['correct'] / values['n']) for (exp, ds), values in sorted(counts.items())]
    if sum(g['n'] for g in groups) != 9262:
        raise RuntimeError('Final score count mismatch')
    atomic_json(OUT / 'summary.json', dict(rows=9262, groups=groups, usage=dict(usage), scoring_signature=scoring_signature))
    with (OUT / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)
    atomic_json(OUT / 'ALL_SCORING_COMPLETE.json', dict(rows=9262, completed_at=time.time(), scoring_signature=scoring_signature, usage=dict(usage)))
    (OUT / 'judge/relay.stop').touch()
    print('SCORING_COMPLETE', 9262, flush=True)


if __name__ == '__main__':
    main()
