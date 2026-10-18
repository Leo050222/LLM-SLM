import json,hashlib
from ablation_core import atomic_json

def migrate(path,target):
 rec=json.loads((path.parent/'recovery_8gpu_20260918.json').read_text())
 assert target['code_sha256']==rec['target_code_sha256']
 assert target['replicas']==8 and target['concurrency']==256 and target['trace_limit_seconds']==900
 source=rec['source_manifests'].get(target['role'])
 if source is not None:
  excluded={'replicas','concurrency','code_sha256','migration'}
  assert {k:v for k,v in source.items() if k not in excluded}=={k:v for k,v in target.items() if k not in excluded},'Scientific configuration changed'
  target['migration']={'source_job':106645,'preserved_completed_and_checkpoints':True,'source_manifest_sha256':hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()}
  assert json.loads(path.read_text()) in (source,target)
 else:assert not path.exists() or json.loads(path.read_text())==target
 atomic_json(path,target)
