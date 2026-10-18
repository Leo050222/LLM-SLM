import json
from queue_rpc import atomic

def migrate(path,target):
 target=json.loads(json.dumps(target))
 assert target['execution']==dict(replicas=1,concurrency=32)
 assert target['runtime_policy']['trace_limit_seconds']==2000
 assert target['settings']['mode']=='weak_resample'
 assert target['settings']['teacher_tokens']==1 and target['settings']['first_token_ban']
 if path.exists():assert json.loads(path.read_text())==target,'Manifest changed'
 else:
  assert not any((path.parent/'completed').glob('*.json'))
  assert not any((path.parent/'traces').glob('*.json'))
  atomic(path,target)
