import json,collections,sys
from pathlib import Path
P=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_baselines/full_teacher_student_x8_20260918')
s=json.loads((P/'summary.json').read_text());configs=json.loads((P/'configs.json').read_text());per=collections.defaultdict(dict)
assert json.loads((P/'ALL_SCORING_COMPLETE.json').read_text())['rows']==73136
for c in configs:
 files=list((P/c['name']/'scores').glob('*.json'));assert len(files)==(2317 if c['family']=='code' else 2254)
 for f in files:
  x=json.loads(f.read_bytes());assert x['scoring_signature']==s['scoring_signature'] and x['role']==c['role'] and x['replicate']==c['replicate'] and type(x['correct']) is bool
  k=(x['role'],x['dataset'],x['problem_id']);assert x['replicate'] not in per[k];per[k][x['replicate']]=x['correct']
 print('Checked',c['name'],file=sys.stderr,flush=True)
hists=collections.defaultdict(lambda:[0]*9)
for (role,ds,pid),v in per.items():
 assert set(v)==set(range(8));hists[role,ds][sum(v.values())]+=1
out=[]
for r in s['combined']:
 h=hists[r['role'],r['dataset']];n=sum(h)
 assert n*8==r['n'] and sum(i*c for i,c in enumerate(h))==r['correct']
 out.append(dict(config=r['role']+'_only',dataset=r['dataset'],questions=n,mean_accuracy=r['accuracy'],pass_at_8_count=n-h[0],pass_at_8=(n-h[0])/n,pass_all_8_count=h[8],pass_all_8=h[8]/n,correct_count_histogram=h))
assert sum(r['questions']*8 for r in out)==73136 and len(out)==16
print(json.dumps(dict(scoring_signature=s['scoring_signature'],results=out)))
