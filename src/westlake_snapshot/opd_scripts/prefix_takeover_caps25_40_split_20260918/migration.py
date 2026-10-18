import json,hashlib
from ablation_core import atomic_json
DEPLOYMENT={'replicas','concurrency','placement','worker_batch_size','teacher_kv_gib','student_kv_gib','code_sha256','migration'}
def scientific(m):return {k:v for k,v in m.items() if k not in DEPLOYMENT}
def migrate(path,target):
 rec=json.loads((path.parent/'switch_split_20260918.json').read_text());source=rec['source_manifest']
 assert target['code_sha256']==rec['target_code_sha256']
 assert scientific(source)==scientific(target),'Scientific configuration changed'
 assert target['replicas']==2 and target['concurrency']==64 and target['worker_batch_size']==8
 target['migration']={'source_job':106421,'source_manifest_sha256':hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest(),'preserve_completed_and_checkpoints':True,'deployment':'split teacher/student GPUs'}
 assert json.loads(path.read_text()) in (source,target),'Unexpected manifest'
 atomic_json(path,target)
