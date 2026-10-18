import asyncio,collections,concurrent.futures,fcntl,hashlib,json,sys,time
from pathlib import Path
import aiohttp
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');G=R/'opd_eval_hard/math_reasoning_grid_x8_20260916';P=R/'opd_rescore/math_grid_score_all_2000_20260919'
sys.path.insert(0,str(R/'opd_scripts/r1_math_score_20260918'));from judge_client import validate
sys.path.insert(0,str(R/'opd_scripts/random_quartile_k1_20260918'));from ablation_core import atomic_json,digest
async def main():
 lock=(G/'scoring.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 failed=json.loads((P/'FAILED.json').read_text());assert len(failed['unresolved'])==1
 dest=Path(failed['unresolved'][0]);v=json.loads(dest.read_text());base=json.loads((P/'RUN_CONFIG.json').read_text())['scoring_signature'];assert v['scoring_signature']==base
 assert hashlib.sha256(Path(v['source']).read_bytes()).hexdigest()==v['source_sha256']
 relay=json.loads((P/'judge/relay.json').read_text());prompt=(P/'judge/system_prompt.txt').read_text();key=(R/'opd_rescore/grid_math_gpt4o_mini_openlux_20260916/.openlux_api_key').read_text().strip()
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120),trust_env=False) as session:
  for budget in [512]:
   async with session.post('https://api.openlux.ai/v1/chat/completions',proxy=f"http://{relay['host']}:{relay['port']}",proxy_auth=aiohttp.BasicAuth('opd',(P/'judge/.relay_token').read_text().strip()),headers={'Authorization':'Bearer '+key},json=dict(model='gpt-4o-mini',temperature=0,max_tokens=budget,frequency_penalty=0.5,stream=False,response_format={'type':'json_object'},messages=[dict(role='system',content=prompt),dict(role='user',content=json.dumps(dict(candidate_answer=v['predicted'],reference_answer=v['reference']),ensure_ascii=False))]),allow_redirects=False) as response:
    data=await response.json();assert response.status==200
   atomic_json(P/f'repair_response_penalty_{budget}.json',data);choice=data['choices'][0];print('REPAIR_RESPONSE',budget,choice['finish_reason'],flush=True)
   if choice['finish_reason']=='stop':decision=validate(choice['message']['content']);break
  else:raise RuntimeError('Judge still truncated; do not accept partial output')
 atomic_json(P/'repair_original_error.json',v)
 signature=digest(dict(base_scoring_signature=base,judge_request_overrides={'max_tokens':budget,'frequency_penalty':0.5}))
 v.update(status='ok',**decision,answer_correct=decision['is_correct'],correct=False if v['finish_reason']=='trace_timeout' else decision['is_correct'],base_scoring_signature=base,scoring_signature=signature,judge_request_overrides={'max_tokens':budget,'frequency_penalty':0.5},repair_reason='Original judge repeats LaTeX until output limit;512/2048/4096 all truncated; repetition penalty0.5 on this row only',api_model=data.get('model'),request_id=data.get('id'),usage=data.get('usage'),judge_raw_content=choice['message']['content'],completed_at=time.time())
 atomic_json(dest,v);atomic_json(P/'REPAIR.json',dict(source=str(dest),base_scoring_signature=base,scoring_signature=signature,max_tokens=budget,frequency_penalty=0.5,at=time.time()))
 cs=json.loads((G/'configs.json').read_text());rows=list(map(json.loads,(G/'prompts.jsonl').read_text().splitlines()))
 def audit(c):
  groups=collections.defaultdict(collections.Counter)
  for row in rows:
   f=G/c['name']/'scores'/f"{row['dataset_index']:05d}.json";x=json.loads(f.read_text());assert type(x['correct']) is bool and type(x['answer_correct']) is bool and x['scoring_signature'] in (base,signature)
   assert x['problem_id']==row['problem_id'] and x['dataset']==row['dataset'] and x['dataset_index']==row['dataset_index']
   if x['scoring_signature']==signature:assert f==dest
   z=groups[x['dataset']];z['n']+=1;z['correct']+=x['correct'];z['answer_correct']+=x['answer_correct'];z['timeouts']+=x['finish_reason']=='trace_timeout'
  return [dict(config=c['name'],dataset=ds,**z,accuracy=z['correct']/z['n'],answer_accuracy=z['answer_correct']/z['n']) for ds,z in groups.items()]
 summary=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
  for i,result in enumerate(pool.map(audit,cs),1):
   summary.extend(result)
   if i%10==0:print('AUDIT_CONFIGS',i,175,flush=True)
 assert sum(x['n'] for x in summary)==394450
 atomic_json(P/'summary.json',dict(rows=394450,groups=summary,scoring_signature=base,repair_signature=signature,repair_rows=1))
 first=R/'opd_rescore/grid_math_gpt4o_mini_openlux_20260916';assert json.loads((first/'COMPLETE.json').read_text())['rows']==56350
 atomic_json(P/'ALL_SCORING_COMPLETE.json',dict(rows=394450,first_round_rows=56350,all8_rows=450800,completed_at=time.time(),repair_rows=1))
 atomic_json(G/'ALL_SCORING_COMPLETE.json',dict(rows=450800,additional_scores=str(P),first_round_scores=str(first),completed_at=time.time(),repair_rows=1))
 atomic_json(P/'progress.json',dict(processed=394450,total=394450,unresolved=0,complete=True,updated_at=time.time()))
 (P/'FAILED.json').rename(P/'FAILED_resolved_history.json');(P/'judge/relay.stop').touch();print('ALL_SCORING_COMPLETE450800',flush=True)
asyncio.run(main())
