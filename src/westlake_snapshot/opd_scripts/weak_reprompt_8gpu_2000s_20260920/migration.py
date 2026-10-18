"""Allow only the user-authorized resource/deadline migration."""
import json,copy
from queue_rpc import atomic

def migrate(path,target):
 target=json.loads(json.dumps(target))
 assert target['execution']==dict(replicas=4,concurrency=128)
 assert target['runtime_policy']['trace_limit_seconds']==2000
 old=json.loads(path.read_text())
 if old==target:return
 assert old['execution']==dict(replicas=4,concurrency=128)
 assert old['runtime_policy']['trace_limit_seconds']==900
 expected=copy.deepcopy(old)
 expected['execution']=target['execution']
 expected['runtime_policy'].update(trace_limit_seconds=2000,replicas=4,concurrency=128,code_sha256=target['runtime_policy']['code_sha256'])
 assert expected==target,'Unexpected scientific/model/input change during migration'
 archive=path.parent/'manifest.before_8gpu_2000s.json'
 if archive.exists():assert json.loads(archive.read_text())==old
 else:atomic(archive,old)
 atomic(path,target)
