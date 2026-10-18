import json
from ablation_core import atomic_json

def migrate(path,target):
 rec=json.loads((path.parent/'recovery_8gpu_20260921.json').read_text())
 assert target['code_sha256']==rec['target_code_sha256']
 assert (target['replicas'],target['concurrency'],target['trace_limit_seconds'])==(8,256,900)
 source=rec['source_manifests'].get(target['role'])
 if source is not None:
  ignored={'replicas','concurrency','code_sha256'}
  assert {k:v for k,v in source.items() if k not in ignored}=={k:v for k,v in target.items() if k not in ignored}, 'Scientific configuration changed'
  assert json.loads(path.read_text()) in (source,target)
 else:assert not path.exists() or json.loads(path.read_text())==target
 atomic_json(path,target)
