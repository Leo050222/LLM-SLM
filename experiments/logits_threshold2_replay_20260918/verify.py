import json,zipfile,struct,hashlib,math
from pathlib import Path
P=Path(__file__).resolve().parents[2]/'reports/logits_threshold2_replay_20260918/106102'
r=json.loads((P/'result.json').read_text());s=json.loads((P/'sample.json').read_text());trace=json.loads((P/'threshold_trace.json').read_text());ids=json.loads((P/'prefix_token_ids.json').read_text())
assert ids==s['prompt_token_ids']+[x['token_id'] for x in trace[:-1]]
assert len(ids)==r['prefix_length'] and len(trace)-1==r['anchor']['response_position']
assert hashlib.sha256(json.dumps(ids).encode()).hexdigest()==r['prefix_sha256']
assert all(x['signed_gap']<=2 for x in trace[:-1]) and trace[-1]['signed_gap']>2
for x in trace:assert x['signed_gap']==x['student_logprob']-x['teacher_logprob']
def load(n):
 with zipfile.ZipFile(P/(n+'.pt')) as z:b=z.read(next(n for n in z.namelist() if n.endswith('/data/0')))
 assert len(b)==151936*2
 return b,[struct.unpack('<f',struct.pack('<I',i<<16))[0] for i in struct.unpack('<151936H',b)]
ba,a=load('original_decode');bb,b=load('fresh_prefill')
assert ba==load('incremental_repeat')[0] and bb==load('fresh_prefill_repeat')[0]
d=[abs(x-y) for x,y in zip(a,b)]
assert max(d)==r['comparison']['max_abs_diff']==1
assert sum(x!=y for x,y in zip(a,b))==r['comparison']['different_elements']==145017
for k in (8,16):
 ia=sorted(range(len(a)),key=lambda i:(-a[i],i))[:k];ib=sorted(range(len(b)),key=lambda i:(-b[i],i))[:k]
 assert ia==r['comparison'][f'top{k}']['original_ids'] and ib==r['comparison'][f'top{k}']['replay_ids']
(P/'verification.json').write_text(json.dumps(dict(passed=True,checks=['prefix excludes trigger','earliest strict signed threshold >2','full tensor differences','both controls bitwise equal','top8/top16 ordering'],sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in P.iterdir() if p.suffix in ('.pt','.json')}),indent=2))
print('Independent artifact verification passed')
