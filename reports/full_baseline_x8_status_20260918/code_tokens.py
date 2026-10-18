import json,collections,concurrent.futures
from pathlib import Path
P=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_baselines/full_teacher_student_x8_20260918')
cs=[c for c in json.loads((P/'configs.json').read_text()) if c['family']=='code']
def scan(c):
 totals=collections.defaultdict(collections.Counter)
 fs=list((P/c['name']/'completed').glob('*.json'));assert len(fs)==2317
 for f in fs:
  x=json.loads(f.read_bytes());n=len(x['response_token_ids']);assert len(x['token_sources'])==n and x['status']=='COMPLETE'
  t=x['token_sources'].count('teacher');assert t==(n if c['role']=='teacher' else 0)
  assert not any(e.get('bridge_token_ids') for e in x['events'])
  totals[x['dataset']].update(n=1,output_tokens=n,teacher_tokens=t)
 return dict(role=c['role'],replicate=c['replicate'],totals=dict(totals))
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:out=list(pool.map(scan,cs))
assert sum(v['n'] for g in out for v in g['totals'].values())==37072
print(json.dumps(out))
