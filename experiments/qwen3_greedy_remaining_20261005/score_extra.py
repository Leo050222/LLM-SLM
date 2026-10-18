"""Score the six remaining math/code datasets with the repository scorers."""
import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

R = Path('/storage/yukaichengLab/wangxin/test_eiloeo')
P = R / 'opd_eval_baselines/qwen3_greedy_remaining_20261005'
sys.path.insert(0, str(R / 'eval_repos/evalscope'))
sys.path.insert(0, str(Path(__file__).parent))
from code_scorer import extract, sandbox

EXPECTED = {
    'math500': 500,
    'cruxeval_o': 800,
    'cruxeval_i': 800,
    'humaneval_plus': 164,
    'mbpp_plus': 378,
    'livecodebench': 175,
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_code_data():
    data_dir = P / 'data'
    data = {}
    for ds, name, key in [
        ('cruxeval_o', 'cruxeval.jsonl', 'id'),
        ('cruxeval_i', 'cruxeval.jsonl', 'id'),
        ('humaneval_plus', 'HumanEvalPlus.jsonl', 'task_id'),
        ('mbpp_plus', 'MbppPlus.jsonl', 'task_id'),
        ('livecodebench', 'livecodebench_v6_incremental.jsonl', 'question_id'),
    ]:
        items = [json.loads(line) for line in (data_dir / name).read_text().splitlines()]
        data[ds] = {
            (item['platform'] + '/' + item[key] if ds == 'livecodebench' else item[key]): item
            for item in items
        }
    return data


def score_math(row):
    from evalscope.metrics.math.parser import extract_answer, math_equal, strip_answer_string

    pred = None
    status = 'ok'
    try:
        response = row['response'].rsplit('</think>', 1)[-1].strip()
        pred = extract_answer(response)
        ok = bool(math_equal(pred, strip_answer_string(row['solution'])))
        if not pred:
            status = 'missing'
    except Exception as exc:  # preserve a per-example receipt for parser failures
        ok = False
        status = type(exc).__name__
    return dict(correct=bool(ok), predicted=pred, status=status)


def score_one(path, code_data):
    row = json.loads(path.read_text())
    ds = row['dataset']
    result = score_math(row) if ds == 'math500' else None
    if result is None:
        try:
            result = sandbox(dict(
                dataset=ds,
                problem=code_data[ds][row['problem_id']],
                solution=extract(ds, row['response']),
            ))
        except Exception as exc:
            result = dict(correct=False, status='scoring_error:' + type(exc).__name__,
                          error=str(exc)[-500:])
    result.update(
        dataset=ds,
        problem_id=row['problem_id'],
        dataset_index=row['dataset_index'],
        source_sha256=sha256(path),
        tokens=len(row.get('response_token_ids', [])),
        finish_reason=row.get('finish_reason'),
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    assert json.loads((output / 'ALL_GENERATION_COMPLETE.json').read_text())['rows'] == sum(EXPECTED.values())
    files = sorted((output / 'completed').glob('*.json'))
    assert len(files) == sum(EXPECTED.values()), len(files)
    code_data = load_code_data()
    score_dir = output / 'scores'
    score_dir.mkdir(exist_ok=True)

    def action(path):
        dest = score_dir / path.name
        source_hash = sha256(path)
        if dest.exists():
            saved = json.loads(dest.read_text())
            assert saved['source_sha256'] == source_hash
            return saved
        result = score_one(path, code_data)
        tmp = dest.with_suffix('.tmp')
        tmp.write_text(json.dumps(result, ensure_ascii=False))
        tmp.replace(dest)
        return result

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        futures = [pool.submit(action, path) for path in files]
        for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
            results.append(future.result())
            if n % 100 == 0 or n == len(files):
                print('SCORED', n, '/', len(files), flush=True)

    summary = {}
    for ds, expected in EXPECTED.items():
        rows = [row for row in results if row['dataset'] == ds]
        assert len(rows) == expected, (ds, len(rows), expected)
        summary[ds] = {
            'questions': len(rows),
            'correct': sum(bool(row['correct']) for row in rows),
            'accuracy': sum(bool(row['correct']) for row in rows) / len(rows),
            'score_status': dict(Counter(str(row.get('status', 'unknown')) for row in rows)),
            'mean_tokens': sum(row['tokens'] for row in rows) / len(rows),
        }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (output / 'SCORING_COMPLETE').write_text(str(time.time()))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
