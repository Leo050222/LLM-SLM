import json,hashlib,collections
from pathlib import Path
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');P=R/'opd_eval_hard/hard_three_20260911'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert (P/'SCORING_COMPLETE').exists()
meta=load(P/'data_manifest.json');assert sha(P/'prompts.jsonl')==meta['prompts_sha256']
for src in meta['sources']:assert sha(Path(src['path']))==src['sha256']
rows=[json.loads(l) for l in (P/'prompts.jsonl').open()];assert len(rows)==1814 and len({r['problem_id'] for r in rows})==1814
assert collections.Counter(r['dataset'] for r in rows)==dict(aime24=30,aime25=30,olympiad_bench=674,bbh=1080)
assert len(meta['bbh_selected_indices'])==27
for subset,indices in meta['bbh_selected_indices'].items():
 assert len(indices)==len(set(indices))==40
 got=[r['metadata']['source_index'] for r in rows if r['dataset']=='bbh' and r['subset']==subset];assert got==indices
summary=load(P/'summary.json');audit={};solved={};gate=(P/'ALL_GENERATION_COMPLETE.json').stat().st_mtime
assert load(P/'ALL_GENERATION_COMPLETE.json')['rows']==5442
for mode in ['online','student','teacher']:
 assert len(list((P/mode/'completed').glob('*.json')))==1814 and len(list((P/'scores'/mode).glob('*.json')))==1814
 counts=collections.Counter();statuses=collections.Counter();correct=collections.Counter();teachers=0;takeovers=0;judges=0;max_takeovers=0
 solved[mode]=set()
 for r in rows:
  p=P/mode/'completed'/f"{r['dataset_index']:05d}.json";v=load(p);sp=P/'scores'/mode/p.name;s=load(sp)
  assert v['status']=='COMPLETE' and v['problem_id']==r['problem_id']==s['problem_id'] and v['dataset_index']==r['dataset_index']==s['dataset_index']
  assert v['solution']==r['solution'] and v['prompt_token_ids']==r['prompt_token_ids']
  assert s['source_sha256']==sha(p) and sp.stat().st_mtime>=gate and p.stat().st_mtime<=gate
  counts[r['dataset']]+=1;correct[r['dataset']]+=bool(s['correct']);statuses[s['status']]+=1
  if s['correct']:solved[mode].add(r['dataset_index'])
  if mode=='online':
   assert len(v['token_sources'])==len(v['response_token_ids']);teachers+=v['token_sources'].count('teacher');judges+=v.get('judge_calls',0)
   ev=[e for e in v['events'] if 'bridge_token_ids' in e];assert all(0<len(e['bridge_token_ids'])<=8 for e in ev);takeovers+=len(ev);max_takeovers=max(max_takeovers,len(ev))
 for ds,n in counts.items():assert summary[mode][ds]['questions']==n and summary[mode][ds]['correct']==correct[ds]
 audit[mode]=dict(counts=dict(counts),correct=dict(correct),statuses=dict(statuses),teacher_tokens=teachers,takeovers=takeovers,max_takeovers=max_takeovers,judge_calls=judges)
audit['overlap']={b:{'both':len(solved['online']&solved[b]),'online_only':len(solved['online']-solved[b]),'baseline_only':len(solved[b]-solved['online'])} for b in ['student','teacher']}
audit['verified']='5442 rows, all dataset/subset identities and shared prompts, source data/output hashes, temporal all-generation-before-any-score gate, per-intervention teacher tokens<=8'
print(json.dumps(audit,indent=2))
