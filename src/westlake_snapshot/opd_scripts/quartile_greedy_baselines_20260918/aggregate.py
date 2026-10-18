import os,json,csv,hashlib,collections,time
from pathlib import Path
from ablation_core import atomic_json
GROUPS=['k0','k1','k4','k8','k16','k32','teacher_greedy']
def summarize(records):
 groups=collections.defaultdict(list);pairs=collections.defaultdict(dict)
 for x in records:
  assert type(x['correct']) is bool
  for q in ['all',str(x['quartile'])]:
   groups[x['dataset'],x['group'],q].append(x)
   key=x['dataset'],q,x['sample_index'];assert x['group'] not in pairs[key];pairs[key][x['group']]=x['correct']
 summary=[]
 for (ds,g,q),rs in sorted(groups.items()):
  assert len(rs)==(100 if q=='all' else 25)
  summary.append(dict(dataset=ds,group=g,quartile=q,n=len(rs),correct=sum(x['correct'] for x in rs),accuracy=sum(x['correct'] for x in rs)/len(rs),length_limit=sum(x['finish_reason']=='length_limit' for x in rs)))
 changes=[]
 for ds,q in sorted({(ds,q) for ds,q,_ in pairs}):
  ps=[v for (d,quartile,i),v in pairs.items() if d==ds and quartile==q]
  assert all(set(v)==set(GROUPS) for v in ps)
  for g in GROUPS[1:]:
   gains=sum(not v['k0'] and v[g] for v in ps);losses=sum(v['k0'] and not v[g] for v in ps)
   changes.append(dict(dataset=ds,quartile=q,group=g,baseline='k0',n=len(ps),wrong_to_correct=gains,correct_to_wrong=losses,accuracy_delta=(gains-losses)/len(ps)))
 return summary,changes
def main():
 P=Path(os.environ['GRID_ROOT'])
 for marker in ['ALL_SCORING_COMPLETE.json','k0_SCORING_COMPLETE.json','teacher_greedy_SCORING_COMPLETE.json']:assert (P/marker).exists()
 raw=(P/'prompts.jsonl').read_bytes();meta=json.loads((P/'INPUTS_COMPLETE.json').read_text());assert hashlib.sha256(raw).hexdigest()==meta['sha256']
 rows=list(map(json.loads,raw.splitlines()));records=[];signatures=set()
 for g in GROUPS:
  for row in rows:
   name=f"{row['dataset_index']:05d}.json";f=P/g/'completed'/name;source=f.read_bytes();x=json.loads(source);s=json.loads((P/g/'scores'/name).read_text())
   assert s['source_sha256']==hashlib.sha256(source).hexdigest()
   assert x['anchor']==row['anchor'] and x['prompt_token_ids']==row['prompt_token_ids']
   assert x['problem_id']==row['problem_id'] and s['sample_index']==row['dataset_index']
   signatures.add(s['scoring_signature']);records.append(dict(s,group=g))
 assert len(signatures)==1 and len(records)==5600
 summary,changes=summarize(records);assert len(summary)==280 and len(changes)==240
 atomic_json(P/'summary_with_baselines.json',dict(rows=5600,groups=summary,paired_changes=changes,input_sha256=meta['sha256'],scoring_signature=signatures.pop()))
 for name,rs in [('summary_with_baselines.csv',summary),('paired_changes_vs_k0.csv',changes)]:
  with (P/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
 atomic_json(P/'ALL_SEVEN_GROUPS_COMPLETE.json',dict(rows=5600,completed_at=time.time()));print('ALL_SEVEN_GROUPS_COMPLETE',flush=True)
if __name__=='__main__':main()
