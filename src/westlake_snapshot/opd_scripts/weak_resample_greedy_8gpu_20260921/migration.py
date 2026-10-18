import json,copy,time
from queue_rpc import atomic

def migrate(path,target):
 target=json.loads(json.dumps(target))
 assert target['execution']==dict(replicas=4,concurrency=128)
 assert target['runtime_policy']['trace_limit_seconds']==2000
 assert target['settings']['mode']=='weak_resample'
 assert target['settings']['teacher_tokens']==1 and target['settings']['first_token_ban']
 assert path.exists(), 'Resume requires original manifest'
 old=json.loads(path.read_text())
 if old==target:return
 assert old['execution']==dict(replicas=1,concurrency=32)
 a=copy.deepcopy(old);b=copy.deepcopy(target)
 for m in [a,b]:
  m.pop('execution')
  for k in ['replicas','concurrency']:m['runtime_policy'].pop(k)
  for k in ['dispatch.py','migration.py']:m['runtime_policy']['code_sha256'].pop(k)
 assert a==b,'Unexpected scientific or worker source change'
 assert old['runtime_policy']['code_sha256']['dispatch.py']=='6267594f602c0d680f7dab29cecd5bac4d4fdd5f08f8890916853cd449005c1c'
 assert old['runtime_policy']['code_sha256']['migration.py']=='c030fa0b33b4c07b2d5d7cf38068ef0233d73420ecfb89157ce14b1a6ff65612'
 atomic(path.parent/'migration_8gpu_20260921.json',dict(previous=old,target=target,migrated_at=time.time(),completed_preserved=len(list((path.parent/'completed').glob('*.json')))))
 atomic(path,target)
