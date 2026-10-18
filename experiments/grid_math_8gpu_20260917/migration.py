import json, hashlib
from queue_rpc import atomic

def migrate(path, target):
    normalized = json.loads(json.dumps(target)); target.clear(); target.update(normalized)
    rec = json.loads((path.parent.parent / 'recovery_8gpu_20260917/migration.json').read_text())
    assert target['runtime_policy']['code_sha256'] == rec['target_code_sha256']
    assert target['execution'] == {'replicas': 4, 'concurrency': 128}
    assert target['runtime_policy']['trace_limit_seconds'] == 900
    source = rec['source_manifests'].get(path.parent.name)
    if source is None:
        assert not path.exists() or json.loads(path.read_text()) == target
    else:
        excluded = {'execution', 'runtime_policy', 'migration'}
        assert {k:v for k,v in source.items() if k not in excluded} == {k:v for k,v in target.items() if k not in excluded}, 'Scientific configuration changed'
        target['migration'] = {'version':'8gpu_20260917', 'source_manifest_sha256':hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest(), 'preserved_completed_and_timeouts':True}
        assert json.loads(path.read_text()) in (source, target), 'Unexpected source manifest'
    atomic(path, target)
