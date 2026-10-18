"""Initialize this independent experiment, or strictly validate its manifest."""
import json
from queue_rpc import atomic


def migrate(path, target):
    normalized = json.loads(json.dumps(target))
    target.clear()
    target.update(normalized)
    assert target['execution'] == {'replicas': 4, 'concurrency': 128}
    assert target['runtime_policy']['trace_limit_seconds'] == 2000
    assert target['settings']['mode'] == 'weak_reprompt'
    assert target['settings']['teacher_tokens'] == 1
    if path.exists():
        assert json.loads(path.read_text()) == target, 'Experiment manifest changed; explicit reviewed migration required'
    else:
        assert not any((path.parent / 'completed').glob('*.json')), 'Completed outputs without manifest'
        assert not any((path.parent / 'traces').glob('*.json')), 'Checkpoints without manifest'
        atomic(path, target)
