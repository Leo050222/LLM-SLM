import json,hashlib
from queue_rpc import atomic

def migrate(path,target):
 normalized=json.loads(json.dumps(target));target.clear();target.update(normalized)
 rec=json.loads((path.parent.parent/'recovery_colocated16_20260918/migration.json').read_text())
 assert target['runtime_policy']['code_sha256']==rec['target_code_sha256']
 assert target['execution']=={'replicas':8,'concurrency':256} and target['runtime_policy']['nodes']==2
 assert target['runtime_policy']['trace_limit_seconds']==900
 source=rec['source_manifests'][path.parent.name]
 excluded={'execution','runtime_policy','migration'}
 assert {k:v for k,v in source.items() if k not in excluded}=={k:v for k,v in target.items() if k not in excluded},'Scientific configuration changed'
 target['migration']={'version':'colocated16_20260918','source_manifest_sha256':hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest(),'preserved_completed_and_timeouts':True,'disjoint_config_shard':target['runtime_policy']['shard']}
 assert json.loads(path.read_text()) in (source,target),'Unexpected manifest'
 atomic(path,target)
