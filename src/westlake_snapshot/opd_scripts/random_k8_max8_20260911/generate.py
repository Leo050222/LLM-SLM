import os,sys,json,hashlib,random,fcntl
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');P=R/'opd_eval_hard/random_k8_max8_20260911';C=R/'opd_online_code/20260911_chunk64_k8_2nodes/ocrd_pipeline'
sys.path.insert(0,str(C))
from online_rollout.core import atomic_json,FileRPC,digest
from online_rollout.run import Backend
N=2314;SEED=20260911

def positions(length,index):
 # Uniform over feasible k-subsets of token offsets with minimum gap 8.
 k=min(8,(length+7)//8)
 rng=random.Random(f'{SEED}:random_anchors:{index}')
 return [v+7*j for j,v in enumerate(sorted(rng.sample(range(length-7*(k-1)),k)))] if k else []

def run_one(row,backend,tok):
 i=row['dataset_index'];path=P/'online/traces'/f'{i:05d}.json';out=P/'online/completed'/path.name
 identity=digest(row)
 st=json.loads(path.read_text()) if path.exists() else dict(identity=identity,status='RUNNING',response_token_ids=[],token_sources=[],events=[],rounds=0,teacher_count=0,judge_calls=0)
 assert st['identity']==identity
 limit=min(40264,40960-len(row['prompt_token_ids'])-1)
 while st['status']!='COMPLETE':
  ids=st['response_token_ids'];offset=len(ids)
  if offset>=limit:st.update(status='COMPLETE',finish_reason='length');break
  anchors=row['random_anchor_offsets'];j=st['teacher_count'];prefix=row['prompt_token_ids']+ids
  seed=int(digest([SEED,i,st['rounds']])[:8],16)%(2**31)
  event=dict(offset=offset)
  if j<len(anchors) and offset==anchors[j]:
   draft=backend.generate('student',prefix,1,seed)
   assert len(draft['token_ids'])==1
   if draft['finish_reason']=='stop':res=draft;role='student'
   else:
    res=backend.generate('teacher',prefix,min(8,limit-offset),seed,banned_first=draft['token_ids'][0]);role='teacher'
    assert res['token_ids'][0]!=draft['token_ids'][0]
    assert len(res['token_ids'])<=8
    st['teacher_count']+=1
    event.update(anchor_response_position=offset,bridge_token_ids=res['token_ids'],student_entry_token=draft['token_ids'][0])
  else:
   assert j>=len(anchors) or offset<anchors[j]
   n=min(limit-offset,anchors[j]-offset) if j<len(anchors) else limit-offset
   res=backend.generate('student',prefix,n,seed);role='student'
  assert res['token_ids']
  st['response_token_ids']+=res['token_ids'];st['token_sources']+=[role]*len(res['token_ids']);st['events'].append(event);st['rounds']+=1
  if res['finish_reason']=='stop':st.update(status='COMPLETE',finish_reason='stop')
  atomic_json(path,st)
 atomic_json(path,st)
 atomic_json(out,dict(row,**st,response=tok.decode(st['response_token_ids'],skip_special_tokens=True)))
 return i

def main():
 (P/'online').mkdir(exist_ok=True);lock=(P/'online/coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 rows=[json.loads(l) for l in (P/'prompts.jsonl').open()];assert len(rows)==N
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained('/storage/yukaichengLab/share/model/Qwen3-1.7B',local_files_only=True)
 bs=[]
 for i in range(8):
  q=P/f'queue/replica_{i}';models={role:json.loads((q/role/'model.json').read_text()) for role in ['student','teacher']};bs.append(Backend(FileRPC(q,1800),models,tok))
 manifest=dict(rows=N,prompt_sha256=hashlib.sha256((P/'prompts.jsonl').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),models=models,temperature=1,top_p=1,seed=SEED,thinking=False,teacher_tokens=8,max_takeovers=8,position_policy='uniform feasible nonoverlapping 8-offset subset over prior student-only response length, early EOS stops',first_token_ban=True,judge=False,max_tokens=40264)
 mp=P/'manifest.json'
 if mp.exists():assert json.loads(mp.read_text())==manifest
 else:atomic_json(mp,manifest)
 failures=[]
 with ThreadPoolExecutor(max_workers=128) as pool:
  pending={pool.submit(run_one,r,bs[r['dataset_index']%8],tok):r['dataset_index'] for r in rows}
  for k,f in enumerate(as_completed(pending),1):
   try:f.result()
   except Exception as e:failures.append(dict(index=pending[f],error=type(e).__name__));print('failed',failures[-1],flush=True)
   if k%100==0:print('settled',k,N,flush=True)
 atomic_json(P/'run_status.json',dict(failed=failures))
 if failures:raise SystemExit(1)
 for r in rows:
  v=json.loads((P/'online/completed'/f"{r['dataset_index']:05d}.json").read_text());assert v['status']=='COMPLETE' and v['problem_id']==r['problem_id'] and v['prompt_token_ids']==r['prompt_token_ids'];assert v['teacher_count']<=8
  ev=[e for e in v['events'] if 'bridge_token_ids' in e];assert len(ev)==v['teacher_count'] and all(len(e['bridge_token_ids'])<=8 for e in ev);assert [e['anchor_response_position'] for e in ev]==r['random_anchor_offsets'][:len(ev)]
 atomic_json(P/'ALL_GENERATION_COMPLETE.json',dict(rows=N,groups=1))
if __name__=='__main__':main()
