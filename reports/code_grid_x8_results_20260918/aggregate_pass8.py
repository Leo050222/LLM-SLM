import json,collections,sys
from pathlib import Path
P=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_code/code_grid_x8_20260916')
s=json.loads((P/'x8_score_summary.json').read_text());configs=sorted({r['config'] for r in s['combined']});out=[]
for c in configs:
 per=collections.defaultdict(dict)
 for rep in range(8):
  files=list((P/f'r{rep}_{c}'/'scores').glob('*.json'));assert len(files)==2317
  for f in files:
   x=json.loads(f.read_bytes());assert x['scoring_signature']==s['scoring_signature'] and x['replicate']==rep and x['base_config']==c and type(x['correct']) is bool
   key=(x['dataset'],x['problem_id']);assert rep not in per[key];per[key][rep]=x['correct']
 hist=collections.defaultdict(lambda:[0]*9)
 for (d,p),v in per.items():
  assert set(v)==set(range(8));hist[d][sum(v.values())]+=1
 for d,h in hist.items():
  n=sum(h);r=next(r for r in s['combined'] if r['config']==c and r['dataset']==d)
  assert n*8==r['n'] and sum(i*n for i,n in enumerate(h))==r['correct']
  out.append(dict(config=c,dataset=d,questions=n,correct_count_histogram=h,pass_at_8_count=n-h[0],pass_all_8_count=h[8],pass_at_8=(n-h[0])/n,pass_all_8=h[8]/n,mean_accuracy=r['accuracy']))
 print('Aggregated',c,file=sys.stderr,flush=True)
assert len(out)==125
print(json.dumps(dict(scoring_signature=s['scoring_signature'],results=out)))
