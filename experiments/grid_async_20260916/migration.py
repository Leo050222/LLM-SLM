import json,hashlib
from pathlib import Path
from queue_rpc import atomic

def hashes(root):
 return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')}

def compatible(old,new):
 for k in set(old)|set(new):
  if k not in ('models','runtime_policy','migration') and old.get(k)!=new.get(k):
   raise ValueError('Scientific configuration changed: '+k)
 for role in ['student','teacher']:
  a,b=old['models'][role],new['models'][role]
  assert a['protocol'] in (1,2) and b['protocol']==3
  assert a['worker_sha256'] in ('63f31492dbeea547b0917c4ee8525b8a9aa22caecc2fae9c66e5061ff99891b4','df2f5f7c44e7e4c88c1fda8cbb846d747a3353efca60765752ddc9c39d9b38a2')
  assert {k:v for k,v in a.items() if k not in ('protocol','worker_sha256')}=={k:v for k,v in b.items() if k not in ('protocol','worker_sha256')}

def migrate(path,target):
 normalized=json.loads(json.dumps(target));target.clear();target.update(normalized)
 rec=json.loads((path.parent.parent/'recovery_async_v3/migration.json').read_text())
 assert target['runtime_policy']['code_sha256']==rec['target_code_sha256']
 assert all(v['worker_sha256']==rec['target_worker_identity'] for v in target['models'].values())
 if not path.exists():
  assert path.parent.name not in rec['source_manifests']
  atomic(path,target);return
 old=json.loads(path.read_text());source=rec['source_manifests'][path.parent.name]
 compatible(source,target)
 target['migration']={'version':'async_v3_20260916','source_manifest_sha256':hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest(),'preserved_completed_and_timeouts':True}
 if old!=target:
  assert old==source, 'Unapproved source manifest'
  atomic(path,target)
