import collections, hashlib, json
from pathlib import Path
from ablation_core import atomic_json,REPROMPT_PROTOCOL
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo')
P=R/'opd_eval_ablation/weak_reprompt_clean_greedy_t2_k1_20260921'
def main():
 P.mkdir(parents=True,exist_ok=True)
 assert not (P/'INPUTS_COMPLETE.json').exists(), 'Inputs already frozen'
 rows=[]; manifests=[]; sources=[]
 for root in [R/'opd_eval_code/code_grid_x8_20260916',R/'opd_eval_hard/math_reasoning_grid_x8_20260916']:
  m=json.loads((root/'r1_k1_t2/manifest.json').read_text());manifests.append(m)
  raw=(root/'prompts.jsonl').read_bytes();sources.append(dict(path=str(root/'prompts.jsonl'),sha256=hashlib.sha256(raw).hexdigest()))
  for row in map(json.loads,raw.decode().splitlines()):
   new=dict(row,original_dataset_index=row['dataset_index']);new['dataset_index']=len(rows);rows.append(new)
 assert len(rows)==4571 and len({(r['dataset'],r['problem_id']) for r in rows})==4571
 settings=dict(manifests[0]['settings']);settings.update(mode='weak_reprompt',teacher_tokens=1,gap_threshold=2.,max_takeovers=None,random_rates=[])
 assert all({k:v for k,v in m['settings'].items() if k!='random_rates'}=={k:v for k,v in manifests[0]['settings'].items() if k!='random_rates'} for m in manifests)
 raw=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode();(P/'prompts.jsonl').write_bytes(raw)
 atomic_json(P/'source_manifests.json',manifests);atomic_json(P/'base_settings.json',settings)
 atomic_json(P/'configs.json',[dict(name='r1_weak_reprompt_clean_t2_k1',k=1,threshold=2,seed=settings['seed'],replicate_index=1,base_config='weak_reprompt_clean_t2_k1')])
 meta=dict(rows=len(rows),sha256=hashlib.sha256(raw).hexdigest(),counts=dict(collections.Counter(r['dataset'] for r in rows)),sources=sources,main_model='Qwen3-1.7B',intervention_model='Qwen3-1.7B',scoring_model='Qwen3-14B',rollouts=1,reprompt_protocol=REPROMPT_PROTOCOL)
 atomic_json(P/'INPUTS_COMPLETE.json',meta);print(json.dumps(meta))
if __name__=='__main__':main()
