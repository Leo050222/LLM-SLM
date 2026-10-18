"""Compact human-readable result from exact per-position comparisons."""
import csv,collections,json
from pathlib import Path
p=Path(__file__).parent/'results'
with (p/'paired_top1.csv').open() as f:
    rows=list(csv.DictReader(f))
for r in rows:
    r['agree']=r['agree']=='True'
    for key in ['student_pmax','teacher_pmax','student_entropy','teacher_entropy']:
        r[key]=float(r[key])
stats=[]
for label,rr in [('all',rows),('agree',[r for r in rows if r['agree']]),
                 ('disagree',[r for r in rows if not r['agree']]),
                 ('student_ge0.5_teacher_le0.2',[r for r in rows if r['student_entropy']>=.5 and r['teacher_entropy']<=.2]),
                 ('student_ge1_teacher_le0.2',[r for r in rows if r['student_entropy']>=1 and r['teacher_entropy']<=.2])]:
    stats.append(dict(slice=label,n=len(rr),agreement=sum(r['agree'] for r in rr)/len(rr),
                      student_pmax=sum(r['student_pmax'] for r in rr)/len(rr),
                      teacher_pmax=sum(r['teacher_pmax'] for r in rr)/len(rr)))
print(json.dumps(stats,indent=2))
(p/'confidence_slices.json').write_text(json.dumps(stats,indent=2))
with (p/'summary.csv').open() as f:summary=list(csv.DictReader(f))
print('DATASETS')
for r in summary:
    if r['threshold']=='ALL':print(r)
md=['# Top-1 agreement after one teacher token','',
    'Both models predict from identical contexts; raw full-vocabulary argmax at temperature1. All valid k1 intervention positions,8 datasets ×5 thresholds. Terminal positions excluded.','',
    '| Dataset | Positions | Top-1 agreement | Trace-equal agreement | Student mean pmax | Teacher mean pmax |',
    '|---|---:|---:|---:|---:|---:|']
for r in summary:
    if r['threshold']=='ALL' and r['selection']=='all':
        md.append(f"| {r['dataset']} | {r['n']} | {float(r['agreement']):.2%} | {float(r['trace_equal_agreement']):.2%} | {float(r['student_pmax']):.2%} | {float(r['teacher_pmax']):.2%} |")
(p/'RESULTS.md').write_text('\n'.join(md))
