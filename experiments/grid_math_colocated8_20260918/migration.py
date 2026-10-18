import json, hashlib, copy
from queue_rpc import atomic

def scientific(manifest):
    value = copy.deepcopy(manifest)
    for key in ('execution', 'runtime_policy', 'migration'):
        value.pop(key, None)
    for model in value['models'].values():
        model.pop('worker_sha256', None)
    return value

def migrate(path, target):
    normalized = json.loads(json.dumps(target)); target.clear(); target.update(normalized)
    rec = json.loads((path.parent.parent / 'recovery_colocated8_20260918/migration.json').read_text())
    assert target['runtime_policy']['code_sha256'] == rec['target_code_sha256']
    assert target['execution'] == {'replicas': 8, 'concurrency': 256}
    assert target['runtime_policy']['trace_limit_seconds'] == 900
    assert all(m['worker_sha256'] == rec['target_worker_sha256'] for m in target['models'].values())
    source = rec['source_manifests'].get(path.parent.name)
    if source is None:
        assert not path.exists() or json.loads(path.read_text()) == target
    else:
        assert scientific(source) == scientific(target), 'Scientific configuration changed'
        target['migration'] = {'version':'colocated8_20260918', 'source_manifest_sha256':hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest(), 'preserved_completed_and_timeouts':True, 'deployment_change':'8 colocated pairs; batch4; teacher/student KV18/10 GiB'}
        assert json.loads(path.read_text()) in (source, target), 'Unexpected source manifest'
    atomic(path, target)
