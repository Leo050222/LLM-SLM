import json,collections,sys,concurrent.futures
from pathlib import Path
P=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_code/code_grid_x8_20260916')
configs=[dict(name='r0_'+x['config'],base_config=x['config'],source_dir=x['path']) for x in json.loads((P/'replicate0_audit.json').read_text())]+[dict(x,source_dir=str(P/x['name'])) for x in json.loads((P/'configs.json').read_text())]
def scan(c):
 totals=collections.defaultdict(collections.Counter)
 files=list((Path(c['source_dir'])/'completed').glob('*.json')); assert len(files)==2317
 for f in files:
  x=json.loads(f.read_bytes()); sources=x['token_sources']; length=len(x['response_token_ids']); assert len(sources)==length and x['status']=='COMPLETE'
  teacher=sources.count('teacher'); assert teacher==sum(len(e.get('bridge_token_ids',[])) for e in x['events'])
  a=totals[x['dataset']];a.update(n=1,teacher_tokens=teacher,output_tokens=length,interventions=sum(bool(e.get('bridge_token_ids')) for e in x['events']),generation_timeouts=int(x.get('finish_reason')=='trace_timeout'))
 return dict(name=c['name'],config=c['base_config'],datasets=dict(totals))
allgroups=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for g in pool.map(scan,configs):
  allgroups.append(g)
  if len(allgroups)%10==0:print('Groups aggregated:',len(allgroups),'/ 200',file=sys.stderr,flush=True)
assert sum(a['n'] for g in allgroups for a in g['datasets'].values())==463400
print(json.dumps(dict(groups=allgroups)))
