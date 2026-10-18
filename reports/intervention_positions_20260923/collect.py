import json,collections,math,statistics,sys,time
from pathlib import Path
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo')
roots=[('code',R/'opd_eval_code/code_grid_x8_20260916',2317),('math',R/'opd_eval_hard/math_reasoning_grid_x8_20260916',2254)]
allrows=[]; byds=collections.defaultdict(lambda:dict(traces=0,events=0,first=[],allpos=[],firstrel=[],allrel=[],lengths=[],interv=[]))
bycfg=collections.defaultdict(lambda:dict(traces=0,events=0,first=[],allpos=[],firstrel=[],allrel=[],lengths=[],interv=[]))
def add(a,ds,cfg,rep):
 a['traces']+=1; ev=[e for e in x.get('events',[]) if e.get('bridge_token_ids')]
 L=len(x.get('response_token_ids',[])); a['lengths'].append(L);a['interv'].append(len(ev));a['events']+=len(ev)
 if ev:
  p=ev[0]['anchor_response_position'];a['first'].append(p);a['firstrel'].append(p/L if L else None)
  for e in ev:
   q=e['anchor_response_position'];a['allpos'].append(q);a['allrel'].append(q/L if L else None)
for fam,root,nq in roots:
 configs=[]
 if fam=='code':
  configs += [dict(name='r0_'+z['config'],base_config=z['config'],replicate_index=0,source_dir=z['path']) for z in json.loads((root/'replicate0_audit.json').read_text())]
  configs += [dict(c,source_dir=str(root/c['name'])) for c in json.loads((root/'configs.json').read_text())]
 else:
  configs += [dict(name='r0_'+z['config'],base_config=z['config'],replicate_index=0,source_dir=z['path']) for z in json.loads((root/'replicate0_audit.json').read_text())]
  configs += [dict(c,source_dir=str(root/c['name'])) for c in json.loads((root/'configs.json').read_text())]
 for c in configs:
  d=Path(c['source_dir'])/'completed'; files=list(d.glob('*.json'))
  records=list(d.glob('records-*.jsonl'))
  assert sum(1 for rf in records for _ in rf.open())==nq,(fam,c['name'],len(files),records)
  items=[]
  for rf in records: items.extend((lambda z:z.get('data',z))(json.loads(line)) for line in rf.open())
  for x in items:
   ds=x['dataset']; key=(fam,ds); add(byds[key],ds,c['base_config'],c['replicate_index']);add(bycfg[(fam,ds,c['base_config'])],ds,c['base_config'],c['replicate_index'])
print('scanned',sum(v['traces'] for v in byds.values()),'traces',sum(v['events'] for v in byds.values()),'events',file=sys.stderr)
def q(v,p):
 v=sorted(x for x in v if x is not None)
 if not v:return None
 z=(len(v)-1)*p;i=math.floor(z);j=math.ceil(z);return v[i]+(v[j]-v[i])*(z-i)
def summ(a):
 def stats(v):return dict(n=len(v),mean=statistics.mean(v) if v else None,median=q(v,.5),p10=q(v,.1),p25=q(v,.25),p75=q(v,.75),p90=q(v,.9),p95=q(v,.95),min=min(v) if v else None,max=max(v) if v else None)
 bins=[0]*10; rbins=[0]*10
 for z in a['allpos']: bins[min(9,int(z//64))]+=1
 for z in a['allrel']: rbins[min(9,int(z*10))]+=1
 return dict(traces=a['traces'],traces_with_intervention=len(a['first']),trace_intervention_rate=len(a['first'])/a['traces'],events=a['events'],mean_events=a['events']/a['traces'],mean_events_conditional=a['events']/len(a['first']) if a['first'] else 0,output_length=stats(a['lengths']),first_position=stats(a['first']),first_relative=stats(a['firstrel']),all_position=stats(a['allpos']),all_relative=stats(a['allrel']),all_position_bins_64=bins,all_relative_decile_bins=rbins,intervention_count_hist=dict(collections.Counter(a['interv'])))
out={'datasets':{f'{fam}/{ds}':summ(a) for (fam,ds),a in byds.items()},'configs':{f'{fam}/{ds}/{cfg}':summ(a) for (fam,ds,cfg),a in bycfg.items()},'meta':dict(position_definition='anchor_response_position: number of final response tokens before Teacher bridge, zero-indexed response boundary; relative position=anchor_response_position/final response token count',bin_definition='absolute bins are [0,64),[64,128),...,[576,640),[640,+); relative bins are deciles [0,.1),...,[.9,1]. Baselines excluded from event positions',created_at=time.time())}
print(json.dumps(out))
