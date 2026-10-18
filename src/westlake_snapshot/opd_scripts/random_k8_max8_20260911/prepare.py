from pathlib import Path
import json,random,hashlib
P=Path(__file__).parent;A=Path('experiments/hard_three_20260911/artifacts');M=Path('experiments/math500_three_20260911/results');rows=[];sources=[]
def add(root,math=False):
 for line in (root/'prompts.jsonl').read_text().splitlines():
  r=json.loads(line);old=r['dataset_index'];f=root/'student/completed'/f'{old:05d}.json';v=json.loads(f.read_text());assert r['prompt_token_ids']==v['prompt_token_ids'];i=len(rows);r['source_dataset_index']=old;r['dataset_index']=i
  if math:r.update(dataset='math500',subset='math500',problem_id='math500/'+r['problem_id'],metadata={})
  n=len(v['response_token_ids']);k=min(8,(n+7)//8);rng=random.Random(f'20260911:random_anchors:{i}');positions=[x+7*j for j,x in enumerate(sorted(rng.sample(range(n-7*(k-1)),k)))] if k else []
  r.update(reference_student_length=n,reference_student_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),random_anchor_offsets=positions);rows.append(r)
 sources.append(dict(path=str(root/'prompts.jsonl'),sha256=hashlib.sha256((root/'prompts.jsonl').read_bytes()).hexdigest()))
add(A);add(M,True);assert len(rows)==2314 and len({r['problem_id'] for r in rows})==2314
(P/'prompts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));(P/'data_manifest.json').write_text(json.dumps(dict(rows=len(rows),sources=sources,seed=20260911,policy='uniform feasible subset of baseline token offsets separated by at least8',prompts_sha256=hashlib.sha256((P/'prompts.jsonl').read_bytes()).hexdigest()),indent=2))
print('prepared',len(rows))
