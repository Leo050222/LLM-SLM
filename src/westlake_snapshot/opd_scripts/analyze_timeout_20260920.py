from pathlib import Path
from collections import Counter,defaultdict
import json,statistics,time
from transformers import AutoTokenizer
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_ablation')
tok=AutoTokenizer.from_pretrained('/storage/yukaichengLab/share/model/Qwen3-1.7B',local_files_only=True)
def med(a):return round(statistics.median(a),3) if a else None
out={}
for name,job in [('weak_reprompt_t2_k1_20260920',107873),('weak_resample_t2_k1_20260920',107874)]:
 p=R/name;d=p/('r1_'+name.removesuffix('_20260920'));cats=defaultdict(list);examples=[];bans=Counter();model=json.loads((p/'slots'/f'job_{job}'/'queue/replica_0/student/model.json').read_text());eos=model['eos_token_id']
 for f in sorted((d/'completed').glob('*.json')):
  try:x=json.loads(f.read_text())
  except FileNotFoundError:continue
  ids=x['response_token_ids'];events=[e for e in x['events'] if 'bridge_token_ids' in e];tail=ids[-2048:];c=Counter(tuple(tail[i:i+32]) for i in range(max(0,len(tail)-31)));rep=c.most_common(1)[0][1] if c else 0
  row=dict(tokens=len(ids),rounds=x['rounds'],interventions=len(events),elapsed=x.get('trace_elapsed_seconds',0),repeat32=rep,accept_ratio=len(ids)/max(1,x['student_draft_tokens']),last20_interventions=sum('bridge_token_ids' in e for e in x['events'][-20:]),eos_banned=sum(e['student_entry_token']==eos for e in events),eos_in_output=ids.count(eos))
  cats[x['finish_reason']].append(row)
  if x['finish_reason']=='trace_timeout' and len(examples)<4:
   examples.append(dict(id=f.stem,dataset=x['dataset'],**row,first_interventions=[dict(position=e['anchor_response_position'],original=tok.decode([e['student_entry_token']]),new=tok.decode(e['bridge_token_ids']),gap=e['selected_candidate']['signed_logprob_gap_student_minus_teacher']) for e in events[:3]],tail_text=tok.decode(ids[-160:]),last20_draft_lengths=[e['draft_tokens'] for e in x['events'][-20:]]))
 summary={k:dict(n=len(v),**{key:med([z[key] for z in v]) for key in v[0]},repeating_over10=sum(z['repeat32']>=10 for z in v),any_eos_banned=sum(z['eos_banned']>0 for z in v)) for k,v in cats.items()}
 out[str(job)]=dict(summary=summary,examples=examples,eos=eos)
print(json.dumps(out,ensure_ascii=False,indent=2))
