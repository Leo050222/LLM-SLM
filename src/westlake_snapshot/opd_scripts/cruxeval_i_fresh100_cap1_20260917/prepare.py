import collections, hashlib, json, random
from pathlib import Path
from ablation_core import atomic_json,digest
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo')
P=R/'opd_eval_prefix/cruxeval_i_fresh100_cap1_20260917'
OLD=R/'opd_eval_prefix/prefix_takeover_caps_20260917'
SOURCES=[R/'opd_eval_code/code_grid_x8_20260916',R/'opd_eval_hard/math_reasoning_grid_x8_20260916']
SEED=2026091711

def main():
 P.mkdir(parents=True,exist_ok=True)
 if (P/'INPUTS_COMPLETE.json').exists():
  m=json.loads((P/'INPUTS_COMPLETE.json').read_text());assert hashlib.sha256((P/'prompts.jsonl').read_bytes()).hexdigest()==m['sha256'];print('INPUTS_ALREADY_FROZEN');return
 old_raw=(OLD/'prompts.jsonl').read_bytes()
 old_rows=list(map(json.loads,old_raw.splitlines()))
 excluded={r['problem_id'] for r in old_rows if r['dataset']=='cruxeval_i'}
 assert len(excluded)==100
 pools=collections.defaultdict(list)
 for root in SOURCES:
  manifest=json.loads((root/'r1_k1_t2/manifest.json').read_text())
  assert manifest['settings']['gap_threshold']==2 and manifest['settings']['teacher_tokens']==1
  for row in map(json.loads,(root/'prompts.jsonl').read_text().splitlines()):
   if row['dataset']=='cruxeval_i' and row['problem_id'] not in excluded:pools[row['dataset']].append((root,row))
 assert set(pools)=={'cruxeval_i'}
 selected=[];audit=[];counts={}
 for ds,pool in sorted(pools.items()):
  rng=random.Random(digest([SEED,ds]));rng.shuffle(pool);n=0
  for rank,(root,row) in enumerate(pool):
   f=root/'r1_k1_t2/completed'/f"{row['dataset_index']:05d}.json";raw=f.read_bytes();x=json.loads(raw)
   assert x['status']=='COMPLETE' and x['problem_id']==row['problem_id'] and x['prompt_token_ids']==row['prompt_token_ids']
   ev=next((e for e in x['events'] if 'bridge_token_ids' in e),None)
   entry={'dataset':ds,'problem_id':row['problem_id'],'draw_rank':rank,'source':str(f),'source_sha256':hashlib.sha256(raw).hexdigest(),'source_finish_reason':x['finish_reason']}
   if ev is None:
    entry['decision']='skip_no_observed_anchor';audit.append(entry);continue
   a=ev['anchor_response_position'];prefix=x['response_token_ids'][:a]
   assert len(prefix)==a and all(t=='student' for t in x['token_sources'][:a])
   assert ev['selected_candidate']['signed_logprob_gap_student_minus_teacher']>2
   assert ev['requested_teacher_tokens']==1 and x['judge_calls']==0
   new=dict(row);new['original_dataset_index']=row['dataset_index'];new['dataset_index']=len(selected)
   new['anchor']={'response_prefix_token_ids':prefix,'frozen_prefix_sha256':digest(row['prompt_token_ids']+prefix),'entry_token':ev['student_entry_token'],'candidate':ev['selected_candidate'],'source':str(f),'source_sha256':entry['source_sha256'],'source_round':ev['round'],'source_response_position':a,'source_seed':20260912}
   selected.append(new);entry['decision']='selected';entry['sample_index']=new['dataset_index'];audit.append(entry);n+=1
   if n==100:break
  assert n==100,(ds,n);counts[ds]=n
 assert len(selected)==100 and len({r['problem_id'] for r in selected})==100
 assert not ({r['problem_id'] for r in selected}&excluded)
 assert not ({r['anchor']['frozen_prefix_sha256'] for r in selected}&{r['anchor']['frozen_prefix_sha256'] for r in old_rows})
 raw=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected).encode();(P/'prompts.jsonl').write_bytes(raw)
 atomic_json(P/'sampling_audit.json',audit)
 atomic_json(P/'INPUTS_COMPLETE.json',{'rows':100,'excluded_previous_problem_ids':sorted(excluded),'previous_input_sha256':hashlib.sha256(old_raw).hexdigest(),'counts':counts,'sha256':hashlib.sha256(raw).hexdigest(),'sampling_seed':SEED,'branch_seeds':[2026091702+i for i in range(8)],'caps':[1],'expected_outputs':800,'anchor_source':'existing r1_k1_t2 trajectory, before any teacher token; uniformly shuffled problem pool with rejection of unobserved anchors','selection_caveat':'conditioned on observed anchor in source rollout; no-anchor censored trajectories are recorded separately'})
 print('INPUTS_FROZEN',counts,'skipped',len(audit)-100,flush=True)
if __name__=='__main__':main()
