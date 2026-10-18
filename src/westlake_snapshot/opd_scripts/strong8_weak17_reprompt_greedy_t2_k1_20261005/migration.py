"""Create this independent experiment's manifest; reject mismatched resumes."""
import json
from queue_rpc import atomic


def migrate(path, target):
    normalized = json.loads(json.dumps(target))
    target.clear()
    target.update(normalized)
    if path.exists():
        if json.loads(path.read_text()) != target:
            raise ValueError('Existing experiment manifest differs; refusing unsafe resume')
    else:
        atomic(path, target)
