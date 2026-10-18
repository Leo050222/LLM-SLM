import json,re,collections,sys
from pathlib import Path
root=Path(sys.argv[1]); launch=json.loads((root/'launch.json').read_text())
job=launch['gpu_job']; counts=collections.Counter(); bindings={}
models=[]
for rank in range(16):
 text=(root/'logs'/f'{job}_rank{rank}.log').read_text()
 match=re.search(r'WORKER_BINDING rank=(\d+) replica=(\d+) role=(\w+) node=(\S+) visible=(\S+)',text)
 assert match and int(match[1])==rank and int(match[2])==rank//2
 assert match[3]==('student' if rank%2==0 else 'teacher')
 assert len(match[5].split(','))==1
 assert 'CUDA_WITNESS_OK NVIDIA H800' in text
 counts[match[4]]+=1
 engine_pid=int(re.search(r'EngineCore pid=(\d+)',text)[1])
 inventory=json.loads((root/'gpu_inventory'/(match[4]+'.json')).read_text())
 entries=[entry for entry in inventory if entry['pid']==engine_pid]
 assert len(entries)==1 and entries[0]['memory_mib']>10000
 bindings.setdefault(match[4],set()).add(entries[0]['uuid'])
 model=json.loads((root/'queue'/f'replica_{rank//2}'/match[3]/'model.json').read_text())
 models.append(model)
assert counts=={'gnho011':8,'gnho034':8},counts
assert all(len(ids)==8 for ids in bindings.values()),bindings
assert all(model==models[i%2] for i,model in enumerate(models))
migration=json.loads((root/'results/migration.json').read_text())
source=Path(migration['source_run'])
for path in (source/'results/completed').glob('*.json'):
 old=json.loads(path.read_text()); new=json.loads((root/'results/completed'/path.name).read_text())
 assert old['response_token_ids']==new['response_token_ids']
 assert old['token_sources']==new['token_sources']
manifest=json.loads((root/'results/manifest.json').read_text())
assert manifest['execution']=={'replicas':8,'concurrency':256}
assert manifest['multiple_takeovers'] and launch['takeover_count_limit'] is None
value=dict(status='TWO_NODE_16GPU_VERIFIED',nodes=dict(counts),replicas=8,concurrency=256,
           preserved_completed=migration['counts']['completed'],migrated_checkpoints=migration['counts']['traces'])
(root/'scale_verification.json').write_text(json.dumps(value,indent=2))
print(json.dumps(value,indent=2))
