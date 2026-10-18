"""Freeze precisely the six missing benchmark inputs and their test data."""
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

R = Path('/storage/yukaichengLab/wangxin/test_eiloeo')
P = R / 'opd_eval_baselines/qwen3_greedy_remaining_20261005'
EXPECTED = dict(math500=500, cruxeval_o=800, cruxeval_i=800,
                humaneval_plus=164, mbpp_plus=378, livecodebench=175)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    P.mkdir(parents=True, exist_ok=True)
    sources = [R/'opd_eval_math500/math500_three_20260911/prompts.jsonl',
               R/'opd_eval_code/code_reasoning_grid_20260913/prompts.jsonl']
    rows = []
    for source in sources:
        for line in source.read_text().splitlines():
            row = json.loads(line)
            row.setdefault('dataset', 'math500')
            row['source_dataset_index'] = row['dataset_index']
            row['dataset_index'] = len(rows)
            rows.append(row)
    assert Counter(r['dataset'] for r in rows) == EXPECTED
    assert len({(r['dataset'], r['problem_id']) for r in rows}) == 2817
    blob = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows)
    dest = P/'prompts.jsonl'
    if dest.exists():
        assert dest.read_text() == blob, 'Frozen prompt mismatch'
    else:
        dest.write_text(blob)
    (P/'data').mkdir(exist_ok=True)
    for name in ['cruxeval.jsonl', 'HumanEvalPlus.jsonl', 'MbppPlus.jsonl',
                 'livecodebench_v6_incremental.jsonl']:
        source = R/'opd_eval_code/code_reasoning_grid_20260913/data'/name
        target = P/'data'/name
        if target.exists():
            assert digest(source) == digest(target)
        else:
            shutil.copyfile(source, target)
    manifest = dict(rows=len(rows), datasets=EXPECTED, thinking=False,
                    prompt_sha256=digest(dest),
                    sources={str(s):digest(s) for s in sources},
                    data_sha256={p.name:digest(p) for p in (P/'data').iterdir()})
    (P/'INPUTS_COMPLETE.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2),flush=True)

if __name__ == '__main__':
    main()
