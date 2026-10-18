"""Fail-closed, explicitly recorded performance-only manifest migration."""
import json
from pathlib import Path


def compatible(source, target):
    for key in set(source) | set(target):
        if key not in ('models', 'code_sha256', 'migration'):
            if source.get(key) != target.get(key):
                raise ValueError('Scientific manifest changed: ' + key)
    for role in ('student', 'teacher'):
        a, b = source['models'][role], target['models'][role]
        for key in set(a) | set(b):
            if key not in ('protocol', 'worker_sha256') and a.get(key) != b.get(key):
                raise ValueError('Model identity changed: ' + role + '.' + key)
        if a['protocol'] != 1 or b['protocol'] != 2:
            raise ValueError('Unapproved worker protocol migration')


def migrate_manifest(path, target, atomic_json, digest):
    # Compare persisted JSON representations, including tuple-to-list encoding.
    normalized = json.loads(json.dumps(target))
    target.clear()
    target.update(normalized)
    if not path.exists():
        atomic_json(path, target)
        return
    old = json.loads(path.read_text())
    record_path = path.parent.parent / 'recovery_104008_performance' / 'migration.json'
    record = json.loads(record_path.read_text())
    source = record['source_manifests'][path.parent.name]
    if target['code_sha256'] != record['target_code_sha256']:
        raise ValueError('Migration target code hash differs')
    compatible(source, target)
    target['migration'] = dict(source_manifest_sha256=digest(source),
                               record='recovery_104008_performance/migration.json',
                               preserved_timeouts=True)
    if old != target:
        if old != source:
            raise ValueError('Source manifest differs from archived approved source')
        atomic_json(path, target)

