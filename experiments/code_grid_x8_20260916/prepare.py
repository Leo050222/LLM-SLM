"""Audit old replicate and prepare seven independent code-only grid repeats."""
import hashlib
import json
import sys
from pathlib import Path
from dataclasses import asdict

ROOT = Path('/storage/yukaichengLab/wangxin/test_eiloeo')
sys.path.insert(0, str(ROOT / 'opd_scripts/nojudge_grid_20260912'))
from ablation_core import Settings, digest, atomic_json

old = ROOT / 'opd_eval_code/code_reasoning_grid_20260913'
new = ROOT / 'opd_eval_code/code_grid_x8_20260916'
rows = [json.loads(line) for line in (old / 'prompts.jsonl').read_text().splitlines()]
assert len(rows) == 2317 and len({r['dataset_index'] for r in rows}) == 2317
configs = [c for c in json.loads((old / 'configs.json').read_text()) if 'role' not in c]
assert {(c['k'], c['threshold']) for c in configs} == {(k,t) for k in [1,4,8,16,32] for t in [2,2.5,3,3.5,4]}
assert len(configs) == 25
audit = []
for c in configs:
    d = old / c['name']
    m = json.loads((d / 'manifest.json').read_text())
    expected = Settings(mode='nojudge_grid', max_takeovers=None, teacher_tokens=c['k'], gap_threshold=float(c['threshold']))
    assert m['settings'] == json.loads(json.dumps(asdict(expected)))
    assert m['input_sha256'] == digest(rows)
    assert m['runtime_policy']['trace_limit_seconds'] == 900
    assert json.loads((d / 'VERIFIED.json').read_text())['rows'] == 2317
    assert len(list((d / 'completed').glob('*.json'))) == 2317
    hashes = hashlib.sha256()
    for row in rows:
        f = d / 'completed' / f"{row['dataset_index']:05d}.json"
        raw = f.read_bytes(); x = json.loads(raw)
        assert x['status'] == 'COMPLETE' and x['judge_calls'] == 0
        assert x['identity'] == digest(dict(row=row, settings=asdict(expected)))
        assert x['prompt_token_ids'] == row['prompt_token_ids'] and x['problem_id'] == row['problem_id']
        hashes.update(f.name.encode() + hashlib.sha256(raw).digest())
    audit.append(dict(config=c['name'], rows=2317, ordered_output_digest=hashes.hexdigest(), path=str(d)))
    print('OLD_REPLICATE_VERIFIED', c['name'], flush=True)
new.mkdir(exist_ok=True)
new_configs = [dict(name=f"r{rep}_{c['name']}", base_config=c['name'], k=c['k'], threshold=c['threshold'], replicate_index=rep, seed=20260911+rep) for rep in range(1,8) for c in configs]
assert len(new_configs) == 175
for name, value in [('configs.json', new_configs), ('replicate0_audit.json', audit)]:
    path = new / name
    if path.exists(): assert json.loads(path.read_text()) == value
    else: atomic_json(path, value)
dest = new / 'prompts.jsonl'
if dest.exists(): assert dest.read_bytes() == (old / 'prompts.jsonl').read_bytes()
else: dest.write_bytes((old / 'prompts.jsonl').read_bytes())
atomic_json(new / 'PLAN.json', dict(old_replicate_root=str(old), old_seed=20260911, new_seeds=list(range(20260912,20260919)), new_rows=405475, combined_rows=463400, configs_per_replicate=25, num_replicates=8, max_nodes=1, max_gpus=8, gpu_type='H800', old_results_read_only=True, baseline_repeats=False, timeout_seconds=900))
print('PREPARED', len(new_configs), 'configs', 405475, 'new rows', flush=True)
