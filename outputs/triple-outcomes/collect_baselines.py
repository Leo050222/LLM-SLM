from pathlib import Path
import json
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo')
P=R/'opd_rescore/baseline_math_gpt4o_mini_openlux_20260916'
rows=[]
for line in (P/'inputs.jsonl').read_text().splitlines():
 r=json.loads(line);f=P/'scores'/r['dataset']/r['config']/(Path(r['relative_path']).stem+'.json');s=json.loads(f.read_text())
 assert s['source_sha256']==r['source_sha256'] and s['input_sha256']==r['input_sha256']
 assert type(s['is_correct']) is bool and type(r['old_correct']) is bool
 rows.append(dict(dataset=r['dataset'],config=r['config'],problem_id=str(s['problem_id']),correct=s['is_correct'],old_correct=r['old_correct'],source=str(f)))
C=R/'opd_eval_code/code_reasoning_grid_20260913'
for config in ['student_only','teacher_only']:
 for f in sorted((C/config/'scores').glob('*.json')):
  s=json.loads(f.read_text());assert type(s['correct']) is bool
  rows.append(dict(dataset=s['dataset'],config=config,problem_id=str(s['problem_id']),correct=s['correct'],old_correct=s['correct'],source=str(f)))
assert len(rows)==9142,len(rows)
assert len({(r['dataset'],r['config'],r['problem_id']) for r in rows})==len(rows)
print(json.dumps(rows))
