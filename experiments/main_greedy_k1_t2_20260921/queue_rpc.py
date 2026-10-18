"""FIFO file RPC with per-caller deadline leases and serialized queue mutations."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import time
import uuid


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value))
    tmp.replace(path)


@contextmanager
def locked(base):
    base.mkdir(parents=True, exist_ok=True)
    with (base / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def live(base, key, now):
    leases = base / 'leases' / key
    active = False
    for p in leases.glob('*.json'):
        if json.loads(p.read_text())['expires_at'] > now:
            active = True
        else:
            p.unlink(missing_ok=True)
    return active


class RPC:
    def __init__(self, root, budget):
        self.root, self.budget = root, budget

    def call(self, role, payload):
        self.budget.check()
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        base = self.root / role
        request, result = base / 'requests' / (key + '.json'), base / 'results' / (key + '.json')
        lease = base / 'leases' / key / (uuid.uuid4().hex + '.json')
        started = time.monotonic()
        try:
            with locked(base):
                if not result.exists():
                    atomic(lease, {'expires_at': time.time() + max(0, self.budget.limit - self.budget.elapsed())})
                    # Joining an identical request must not reset its FIFO age.
                    if not request.exists():
                        atomic(request, payload)
            while not result.exists():
                self.budget.check()
                time.sleep(.05)
            self.budget.check()
            value = json.loads(result.read_text())
            if 'error' in value:
                raise RuntimeError(value['error'])
            return dict(value, rpc_wait_seconds=time.monotonic() - started)
        finally:
            with locked(base):
                lease.unlink(missing_ok=True)
                if not live(base, key, time.time()):
                    request.unlink(missing_ok=True)


def claim_batch(base, size):
    """Always include the oldest live request; batch the same op behind it."""
    with locked(base):
        candidates = []
        for p in (base / 'requests').glob('*.json'):
            if (base / 'results' / p.name).exists() or not live(base, p.stem, time.time()):
                p.unlink(missing_ok=True)
                continue
            candidates.append((p.stat().st_mtime_ns, p))
        candidates.sort(key=lambda item: (item[0], item[1].name))
        selected, operation = [], None
        for created, p in candidates:
            req = json.loads(p.read_text())
            if operation is None:
                operation = req['op']
            if req['op'] == operation:
                selected.append((p, req, max(0, time.time() - created / 1e9)))
                if len(selected) == size:
                    break
        return selected


def publish(base, path, value):
    with locked(base):
        atomic(base / 'results' / path.name, value)
        path.unlink(missing_ok=True)
