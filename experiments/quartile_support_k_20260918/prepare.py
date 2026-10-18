import collections, hashlib, json, random, os
from pathlib import Path
from ablation_core import atomic_json, digest
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo')
P=Path(os.environ.get('GRID_ROOT',str(R/'opd_eval_prefix/quartile_support_k_20260918')))
SEED=2026091807
KS=[1,4,8,16,32]
def position(length,quartile,rng):
 lo=quartile*length//4;hi=(quartile+1)*length//4
 assert hi>lo
 return rng.randrange(lo,hi)
def main():
 P.mkdir(parents=True,exist_ok=True)
 if (P/'INPUTS_COMPLETE.json').exists():
  meta=json.loads((P/'INPUTS_COMPLETE.json').read_text())
  assert hashlib.sha256((P/'prompts.jsonl').read_bytes()).hexdigest()==meta['sha256']
  print('INPUTS_ALREADY_FROZEN',meta['counts']);return
 pools=collections.defaultdict(list);manifests={}
 for root in [R/'opd_eval_code/code_grid_x8_20260916',R/'opd_eval_hard/math_reasoning_grid_x8_20260916']:
  raw=(root/'r1_k1_t2/manifest.json').read_bytes();m=json.loads(raw)
  assert m['settings']['teacher_tokens']==1 and m['settings']['gap_threshold']==2
  manifests[str(root)]=hashlib.sha256(raw).hexdigest()
  for row in map(json.loads,(root/'prompts.jsonl').read_text().splitlines()):pools[row['dataset']].append((root,row))
 assert len(pools)==8
 rows=[];audit=[]
 for ds,pool in sorted(pools.items()):
  rng=random.Random(digest([SEED,ds]));chosen=rng.sample(pool,100)
  for j,(root,row) in enumerate(chosen):
   f=root/'r1_k1_t2/completed'/f"{row['dataset_index']:05d}.json";raw=f.read_bytes();x=json.loads(raw)
   assert x['status']=='COMPLETE' and x['problem_id']==row['problem_id'] and x['prompt_token_ids']==row['prompt_token_ids']
   length=len(x['response_token_ids']);q=j//25;a=position(length,q,rng)
   prefix=x['response_token_ids'][:a]
   new=dict(row,original_dataset_index=row['dataset_index'],dataset_index=len(rows),quartile=q+1)
   new['anchor']=dict(response_prefix_token_ids=prefix,prefix_token_sources=x['token_sources'][:a],frozen_prefix_sha256=digest(row['prompt_token_ids']+prefix),source=str(f),source_sha256=hashlib.sha256(raw).hexdigest(),source_finish_reason=x['finish_reason'],source_response_length=length,source_response_position=a,quartile_bounds=[q*length//4,(q+1)*length//4])
   rows.append(new);audit.append(dict(dataset=ds,problem_id=row['problem_id'],sample_index=new['dataset_index'],quartile=q+1,position=a,source_length=length,source_finish_reason=x['finish_reason'],source_sha256=new['anchor']['source_sha256']))
 assert len(rows)==800 and len({(r['dataset'],r['problem_id']) for r in rows})==800
 counts=collections.Counter(f"{r['dataset']}/q{r['quartile']}" for r in rows);assert set(counts.values())=={25}
 raw=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode();(P/'prompts.jsonl').write_bytes(raw)
 atomic_json(P/'sampling_audit.json',audit)
 atomic_json(P/'INPUTS_COMPLETE.json',dict(rows=800,counts=dict(counts),sha256=hashlib.sha256(raw).hexdigest(),seed=SEED,ks=KS,expected_outputs=4000,source_manifests=manifests,source_finish_reasons=dict(collections.Counter(a['source_finish_reason'] for a in audit)),protocol='One teacher generation request of n=32 per prefix, temperature=1; slice common bridge for each k; student temperature=0; no token ban or subsequent intervention'))
 print('INPUTS_FROZEN',dict(counts),flush=True)
if __name__=='__main__':main()
