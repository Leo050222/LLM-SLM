from pathlib import Path
import json,hashlib,time,collections,concurrent.futures,datetime
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');P=R/'opd_eval_code/code_grid_x8_20260916';S=R/'opd_scripts/code_grid_x8_score_20260918'
started=time.time();rows=list(map(json.loads,(P/'prompts.jsonl').read_text().splitlines()));rowmap={r['dataset_index']:r for r in rows}
configs=[dict(name='r0_'+a['config'],base_config=a['config'],replicate_index=0,source_dir=a['path']) for a in json.loads((P/'replicate0_audit.json').read_text())]+[dict(c,source_dir=str(P/c['name'])) for c in json.loads((P/'configs.json').read_text())]
old=Path(json.loads((P/'PLAN.json').read_text())['old_replicate_root'])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
signature=hashlib.sha256(json.dumps(dict(files={n:sha(S/n) for n in ['score.py','code_scorer.py','score_one.py']},data={n:sha(old/'data'/n) for n in ['cruxeval.jsonl','HumanEvalPlus.jsonl','MbppPlus.jsonl','livecodebench_v6_incremental.jsonl']},timeout_policy='source_trace_timeout_count_wrong'),sort_keys=True).encode()).hexdigest()
snap=[(c,list((P/c['name']/'scores').glob('*.json'))) for c in configs]
print('SNAPSHOT',sum(len(fs) for c,fs in snap),flush=True)
def audit(item):
 c,files=item;counts=collections.Counter();statuses=collections.Counter();errors=[];examples={}
 for f in files:
  try:
   x=json.loads(f.read_text());raw=(Path(c['source_dir'])/'completed'/f.name).read_bytes();src=json.loads(raw);row=rowmap[int(f.stem)];ds=row['dataset']
   assert x['source_sha256']==hashlib.sha256(raw).hexdigest(),'source_hash'
   assert x['scoring_signature']==signature,'scoring_signature'
   assert type(x['correct']) is bool,'correct_type'
   assert src['status']=='COMPLETE','source_incomplete'
   for k in ['dataset','problem_id','dataset_index']:assert x[k]==src[k]==row[k],k
   assert src['prompt_token_ids']==row['prompt_token_ids'],'prompt'
   assert x['replicate']==c['replicate_index'] and x['base_config']==c['base_config'],'config'
   status=x['status'];status_key=json.dumps(status,sort_keys=True) if isinstance(status,dict) else status
   if src['finish_reason']=='trace_timeout':assert status=='trace_timeout' and x['correct'] is False and src['timeout_count_as_wrong'],'timeout_policy'
   else:
    assert status!='trace_timeout','spurious_generation_timeout'
    if isinstance(status,dict):assert x['correct']==all(v=='pass' for v in status.values()),'pass_consistency'
    else:assert x['correct']==(status=='pass'),'pass_consistency'
   counts[(ds,'scored')]+=1;counts[(ds,'correct')]+=int(x['correct']);statuses[(ds,status_key)]+=1
   if status in ['scoring_execution_timeout','solution_process_crash'] if isinstance(status,str) else False:
    examples.setdefault(status,[])
    if len(examples[status])<2:examples[status].append(dict(group=c['name'],file=f.name,dataset=ds,status=status,exitcode=x.get('exitcode')))
  except Exception as e:
   if len(errors)<20:errors.append(dict(group=c['name'],file=f.name,error=type(e).__name__,detail=str(e)[:180]))
   counts[('ALL','audit_errors')]+=1
 return counts,statuses,errors,examples
counts=collections.Counter();statuses=collections.Counter();errors=[];examples={}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 for i,(c,s,e,ex) in enumerate(pool.map(audit,snap)):
  counts.update(c);statuses.update(s);errors+=e
  for k,v in ex.items():examples.setdefault(k,[]);examples[k]=(examples[k]+v)[:10]
  if (i+1)%25==0:print('AUDITED_GROUPS',i+1,flush=True)
report=dict(snapshot_started_at=started,finished_at=time.time(),scoring_signature=signature,expected_scores=463400,snapshot_score_files=sum(len(fs) for c,fs in snap),audit_errors=counts[('ALL','audit_errors')],errors=errors[:50],datasets={ds:dict(scored=counts[(ds,'scored')],correct=counts[(ds,'correct')],expected=sum(r['dataset']==ds for r in rows)*200,statuses={s:n for (d,s),n in statuses.items() if d==ds}) for ds in sorted({r['dataset'] for r in rows})},exception_examples=examples)
f=P/'parallel_scoring_20260918'/('AUDIT_'+str(int(started))+'.json');f.write_text(json.dumps(report,indent=2));print('REPORT',f);print(json.dumps(report))
