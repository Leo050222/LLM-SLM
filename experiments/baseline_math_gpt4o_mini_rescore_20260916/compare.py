import collections,csv,json
from pathlib import Path
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_rescore')
P=R/'baseline_math_gpt4o_mini_openlux_20260916'
G=R/'grid_math_gpt4o_mini_openlux_20260916'
assert json.loads((P/'COMPLETE.json').read_text())['rows']==4508
c=collections.defaultdict(collections.Counter)
for line in (P/'inputs.jsonl').read_text().splitlines():
 r=json.loads(line);s=json.loads((P/'scores'/r['dataset']/r['config']/(Path(r['relative_path']).stem+'.json')).read_text())
 assert s['source_sha256']==r['source_sha256'] and s['input_sha256']==r['input_sha256']
 assert isinstance(s['is_correct'],bool)
 v=c[(r['dataset'],r['config'])];a=r['old_correct'];b=s['is_correct'];v['n']+=1;v['old_correct']+=a;v['new_correct']+=b
 v['wrong_to_right']+=(not a and b);v['right_to_wrong']+=(a and not b)
baseline=[dict(dataset=ds,config=g,**v,old_accuracy_percent=100*v['old_correct']/v['n'],new_accuracy_percent=100*v['new_correct']/v['n'],delta_pp=100*(v['new_correct']-v['old_correct'])/v['n']) for (ds,g),v in sorted(c.items())]
(P/'baseline_comparison.json').write_text(json.dumps(baseline,indent=2)+'\n')
with (P/'baseline_comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(baseline[0]));w.writeheader();w.writerows(baseline)
groups=json.loads((G/'summary.json').read_text())['groups']+json.loads((P/'summary.json').read_text())['groups']
combined=[]
for row in groups:
 ds=row['dataset'];n=row['total'];acc=100*row.get('correct',0)/n
 bs={g:100*c[(ds,g)]['new_correct']/c[(ds,g)]['n'] for g in ['student_only','teacher_only']}
 combined.append(dict(dataset=ds,config=row['config'],n=n,correct=row.get('correct',0),accuracy_percent=acc,delta_pp_vs_student=acc-bs['student_only'],delta_pp_vs_teacher=acc-bs['teacher_only']))
with (P/'grid_with_baselines.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(combined[0]));w.writeheader();w.writerows(combined)
print('BASELINE_COMPARISON',json.dumps(baseline),flush=True)
