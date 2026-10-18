"""Read-only field/coverage audit of the frozen main-grid k1_t2 traces."""
import collections
import json
from pathlib import Path

ROOT = Path('/storage/yukaichengLab/wangxin/test_eiloeo')
out = {'groups': [], 'datasets': {}, 'trace_keys': [], 'event_keys': [], 'candidate_keys': []}
datasets = collections.defaultdict(collections.Counter)
for family, rel in [('code', 'opd_eval_code/code_grid_x8_20260916'),
                    ('math', 'opd_eval_hard/math_reasoning_grid_x8_20260916')]:
    root = ROOT / rel
    old = next(x for x in json.loads((root/'replicate0_audit.json').read_text()) if x['config'] == 'k1_t2')
    groups = [(0, Path(old['path']))] + [(c['replicate_index'], root/c['name']) for c in json.loads((root/'configs.json').read_text()) if c['base_config'] == 'k1_t2']
    for replicate, group in groups:
        counts = collections.Counter()
        for path in sorted((group/'completed').glob('*.json')):
            x = json.loads(path.read_text())
            counts['traces'] += 1
            ds = datasets[x['dataset']]
            ds['traces'] += 1
            out['trace_keys'] = sorted(set(out['trace_keys']) | set(x))
            counts['timeouts'] += x.get('finish_reason') == 'trace_timeout'
            has = False
            for e in x['events']:
                if 'bridge_token_ids' not in e:
                    continue
                has = True
                c = e['selected_candidate']; j = e['anchor_response_position']; y = e['bridge_token_ids'][0]
                counts['interventions'] += 1; ds['interventions'] += 1
                out['event_keys'] = sorted(set(out['event_keys']) | set(e))
                out['candidate_keys'] = sorted(set(out['candidate_keys']) | set(c))
                valid = (len(e['bridge_token_ids']) == 1 and x['response_token_ids'][j] == y
                         and x['token_sources'][j] == 'teacher' and e['student_entry_token'] != y
                         and c['student_token_id'] == e['student_entry_token']
                         and c['signed_logprob_gap_student_minus_teacher'] > 2
                         and c['absolute_sequence_position'] == len(x['prompt_token_ids']) + j)
                counts['aligned_events'] += valid
                counts['first_token_ban'] += bool(e.get('first_token_ban'))
                counts['teacher_token_in_saved_teacher_top16'] += any(t['token_id'] == y for t in c.get('teacher_top_tokens', []))
                counts['events_with_student_top_tokens'] += bool(c.get('student_top_tokens'))
                counts['events_with_entropy_field'] += any('entropy' in k for k in list(c) + list(e))
            counts['traces_with_interventions'] += has
        out['groups'].append(dict(family=family, replicate=replicate, source=str(group), **counts))
out['datasets'] = dict(datasets)
out['totals'] = dict(sum((collections.Counter({k:v for k,v in g.items() if isinstance(v,int) and k != 'replicate'}) for g in out['groups']), collections.Counter()))
print(json.dumps(out, indent=2))
