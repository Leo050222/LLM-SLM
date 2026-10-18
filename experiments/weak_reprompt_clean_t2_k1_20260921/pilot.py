"""Bounded actual-GPU pilot; never claims statistical proof of reduced repetition."""
import os,json,hashlib,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from transformers import AutoTokenizer
from ablation_core import Settings,rollout,atomic_json,digest
from reprompt import build_prompt,TEMPLATE_SHA256
from run import Backend
from queue_rpc import RPC
from deadline import Budget
P=Path(os.environ['GRID_ROOT']);slot=Path(os.environ['OPD_ONLINE_RUN']);S=Path(__file__).parent
REPLICAS=int(os.environ['GRID_REPLICAS'])

def periodic_suffix(ids):
    if len(ids)<256:return None
    tail=ids[-256:]
    for period in range(1,33):
        if all(tail[i]==tail[i-period] for i in range(period,len(tail))):return period
    return None

def main():
 rows=list(map(json.loads,(P/'prompts.jsonl').read_text().splitlines()));seen=set();selected=[]
 for row in rows:
  if row['dataset'] not in seen:selected.append(row);seen.add(row['dataset'])
 # Fixed challenge examples chosen before this pilot, from prior diagnosis.
 for i in [5,14,22,218]:
  row=rows[i]
  if row not in selected:selected.append(row)
 assert len(selected)==12 and len(seen)==8
 settings=Settings(mode='weak_reprompt',teacher_tokens=1,max_takeovers=None,max_tokens=1024,seed=20260912)
 atomic_json(slot/'pilot/PLAN.json',dict(rows=[dict(index=r['dataset_index'],dataset=r['dataset']) for r in selected],max_tokens=1024,per_trace_seconds=600,scope='implementation and pathological-repetition screen; not accuracy evaluation',template_sha256=TEMPLATE_SHA256))
 def run_one(pair):
  i,row=pair;q=slot/'queue'/f'replica_{i%REPLICAS}';models={r:json.loads((q/r/'model.json').read_text()) for r in ['student','teacher']}
  tok=AutoTokenizer.from_pretrained(models['student']['model'],local_files_only=True)
  b=Backend(RPC(q,Budget(slot/'pilot/timing'/f"{row['dataset_index']:05d}.json",600)),models,tok)
  normal=b.generate;call=b.call;latest={};witness=[]
  def generate(role,prefix,n,seed,banned_first=None):
   result=normal(role,prefix,n,seed,banned_first);latest.update(prefix=list(prefix),draft=list(result['token_ids']));return result
  def checked_call(role,**kw):
   if kw.get('banned_first') is not None:
    assert role=='student' and kw['n']==1
    # Determine the unique token boundary by exact reconstruction, not decoded strings.
    response=latest['prefix'][len(row['prompt_token_ids']):];draft=latest['draft']
    positions=[j for j,t in enumerate(draft) if t==kw['banned_first'] and build_prompt(tok,row,response,draft,j)==kw['prefix']]
    assert len(positions)==1,'Incorrect corrective context or continuation boundary'
    witness.append(dict(position=positions[0],input_tokens=len(kw['prefix']),ban=kw['banned_first']))
   result=call(role,**kw)
   if kw.get('banned_first') is not None:assert result['token_ids'][0]!=kw['banned_first']
   return result
  b.generate=generate;b.call=checked_call
  snapshots=[]
  def save(x):
   assert len(x['response_token_ids'])==len(x['token_sources'])
   if periodic_suffix(x['response_token_ids']) is not None:
    atomic_json(slot/'pilot'/f"{row['dataset_index']:05d}.json",dict(row=row,state=x,failed='periodic_suffix',response=tok.decode(x['response_token_ids'])))
    raise RuntimeError('Periodic suffix detected; formal submission blocked')
   snapshots.append(len(x['response_token_ids']))
  x=rollout(row,settings,b,save=save);events=[e for e in x['events'] if 'bridge_token_ids' in e]
  assert len(witness)==len(events)
  assert [e['anchor_response_position'] for e in events]==[i for i,v in enumerate(x['token_sources']) if v=='student_reprompt']
  # Check the fresh context contains no marker inserted by our template.
  assert all(e['template_sha256']==TEMPLATE_SHA256 for e in events)
  v=dict(index=row['dataset_index'],dataset=row['dataset'],finish_reason=x['finish_reason'],tokens=len(x['response_token_ids']),interventions=len(events),witness=witness,response=tok.decode(x['response_token_ids']),state=x)
  atomic_json(slot/'pilot'/f"{row['dataset_index']:05d}.json",v)
  print('PILOT_ROW_PASSED',v['index'],v['dataset'],v['finish_reason'],v['tokens'],v['interventions'],flush=True)
  return {k:v[k] for k in ['index','dataset','finish_reason','tokens','interventions']}
 results=[];errors=[]
 # Sequential rows per replica avoid oversubscribing the small validation job.
 def shard(replica):
  local=[]
  for i in range(replica,len(selected),REPLICAS):
   try:local.append(run_one((i,selected[i])))
   except Exception as e:errors.append(dict(index=selected[i]['dataset_index'],type=type(e).__name__,message=str(e)))
  return local
 with ThreadPoolExecutor(max_workers=REPLICAS) as pool:
  for local in pool.map(shard,range(REPLICAS)):results.extend(local)
 report=dict(passed=not errors and len(results)==12,results=results,errors=errors,template_sha256=TEMPLATE_SHA256,code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in S.glob('*.py')},completed_at=time.time(),limits='12 samples, at most 1024 output tokens each; not proof of general accuracy or absence of later repetition')
 atomic_json(slot/'pilot/REPORT.json',report)
 if not report['passed']:raise RuntimeError('Pilot failed; formal job not submitted')
 assert sum(r['interventions'] for r in results)>0
 atomic_json(P/'VALIDATION_PASSED.json',report)
 print('PILOT_PASSED',len(results),flush=True)
if __name__=='__main__':main()
