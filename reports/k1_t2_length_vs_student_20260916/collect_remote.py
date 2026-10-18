from pathlib import Path
import json,hashlib,collections
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo')
BASE={};out=[];samples=collections.defaultdict(list)

def load(p,score):
 blob=p.read_bytes();x=json.loads(blob);s=json.loads(score.read_text())
 assert hashlib.sha256(blob).hexdigest()==s['source_sha256'],str(p)
 assert s['problem_id']==x['problem_id'] and s['dataset_index']==x['dataset_index']
 x['_score']=s;return x

for rel,mode in [('opd_eval_hard/hard_three_20260911','student'),('opd_eval_math500/math500_three_20260911','student'),('opd_eval_code/code_reasoning_grid_20260913','student_only')]:
 root=R/rel
 for p in (root/mode/'completed').glob('*.json'):
  score=root/mode/'scores'/p.name if mode=='student_only' else root/'scores'/mode/p.name
  x=load(p,score);ds=x.get('dataset','math500')
  if ds.startswith('aime'):continue
  pid=x['problem_id'] if ds!='math500' else 'math500/'+x['problem_id']
  assert (ds,pid) not in BASE;BASE[ds,pid]=x
assert len(BASE)==4571

def features(x):
 ids=x['response_token_ids'];text=x['response'];n=len(ids)
 grams=[tuple(ids[i:i+4]) for i in range(max(0,n-3))]
 return {'tokens':n,'finish':x['finish_reason'],'correct':bool(x['_score']['correct']),'score_status':x['_score']['status'],'source_sha256':x['_score']['source_sha256'],'repeat4_fraction':1-len(set(grams))/len(grams) if grams else 0,'input_blocks':text.count('[INPUT]'),'output_blocks':text.count('[OUTPUT]'),'char_length':len(text)}

for rel,g in [('opd_eval_hard/nojudge_grid_20260912','math'),('opd_eval_code/code_reasoning_grid_20260913','code')]:
 root=R/rel
 for p in sorted((root/'k1_t2/completed').glob('*.json')):
  score=root/'scores/k1_t2'/p.name if g=='math' else root/'k1_t2/scores'/p.name
  y=load(p,score);ds=y['dataset'];pid=y['problem_id'];x=BASE[ds,pid]
  assert x['prompt_token_ids']==y['prompt_token_ids'],(ds,pid,'prompt mismatch')
  ev=[e for e in y.get('events',[]) if 'bridge_token_ids' in e]
  assert all(len(e['bridge_token_ids'])==1 for e in ev)
  a,b=features(x),features(y)
  row=dict(dataset=ds,problem_id=pid,k1_dataset_index=y['dataset_index'],student_dataset_index=x['dataset_index'],student=a,k1=b,interventions=len(ev),teacher_tokens=y['token_sources'].count('teacher'),first_intervention=ev[0].get('anchor_response_position')if ev else None)
  out.append(row)
  if a['tokens']-b['tokens']>=100 and b['tokens']<=.5*a['tokens']:
   samples[ds].append((a['tokens']-b['tokens'],dict(row,problem=x['problem'],student_head=x['response'][:1600],student_tail=x['response'][-700:] if len(x['response'])>1600 else '',k1_head=y['response'][:1600],k1_tail=y['response'][-700:]if len(y['response'])>1600 else '',first_event=ev[:1])))
assert len(out)==4571
examples={ds:[v for _,v in sorted(s,key=lambda q:q[0],reverse=True)[:4]] for ds,s in samples.items()}
print(json.dumps({'rows':out,'examples':examples,'audit':{'matched':4571,'source_hashes_verified':9142,'identical_prompt_tokens':4571,'math500_id_mapping':'math500/ prefixed to legacy problem_id','selection':'k1_tokens <= student_tokens/2 AND student_tokens-k1_tokens >=100'}}))
