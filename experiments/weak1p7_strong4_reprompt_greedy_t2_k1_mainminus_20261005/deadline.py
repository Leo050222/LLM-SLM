import json
import threading
import time
import uuid
from pathlib import Path


class TraceDeadline(Exception):
    pass


class Budget:
    def __init__(self, path, limit=900, clock=time.monotonic):
        self.path = Path(path)
        self.clock = clock
        self.limit = limit
        self.previous = json.loads(self.path.read_text())['elapsed_seconds'] if self.path.exists() else 0
        self.start = clock()
        self.last_save = -1e20
        self.lock = threading.Lock()

    def elapsed(self):
        return self.previous + self.clock() - self.start

    def persist(self):
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + '.' + uuid.uuid4().hex + '.tmp')
            tmp.write_text(json.dumps(dict(elapsed_seconds=self.elapsed(), limit_seconds=self.limit,
                                          timing_scope='active trace wall time including RPC waits; excludes scheduler and job downtime')))
            tmp.replace(self.path)
            self.last_save = self.clock()

    def check(self):
        if self.elapsed() >= self.limit:
            self.persist()
            raise TraceDeadline('Trace exceeded cumulative wall-time budget')
        if self.clock() - self.last_save >= 2:
            self.persist()


def consume_tasks(tasks, action, replica):
    import queue
    while True:
        try:
            task = tasks.get_nowait()
        except queue.Empty:
            return
        try:
            action(task, replica)
        finally:
            tasks.task_done()
