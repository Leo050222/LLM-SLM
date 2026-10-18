import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from queue_rpc import RPC, atomic, claim_batch, publish
from deadline import Budget, TraceDeadline
from migration import compatible, migrate_manifest
from entropy_mapping import external_records


class QueueTest(unittest.TestCase):
    def test_entropy_ids_are_not_matched_by_output_order(self):
        x = external_records({'1-abcdef01': {'entropy': .3}, '0-12345678': {'entropy': .9}})
        self.assertEqual(x['0']['entropy'], .9)
        with self.assertRaises(ValueError):
            external_records({'0-abcdef01': {}, '0-12345678': {}})
        with self.assertRaises(ValueError):
            external_records({'unknown': {}})

    def test_fifo_hash_and_expired_leases(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            for key, age, expires, op in [('ff', 100, 100, 'entropy'), ('00', 10, 100, 'generate'), ('01', 200, -1, 'score')]:
                p = base / 'requests' / (key + '.json')
                atomic(p, {'op': op})
                os.utime(p, (time.time() - age,) * 2)
                atomic(base / 'leases' / key / 'a.json', {'expires_at': time.time() + expires})
            batch = claim_batch(base, 8)
            self.assertEqual([x[0].stem for x in batch], ['ff'])
            self.assertFalse((base / 'requests/01.json').exists())
            publish(base, batch[0][0], {'ok': True})
            self.assertEqual(claim_batch(base, 8)[0][0].stem, '00')

    def test_shared_request_timeout_does_not_cancel_other_waiter(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); base = root / 'student'; results = []
            def call(limit, name):
                try:
                    results.append((name, RPC(root, Budget(root / (name + '.json'), limit)).call('student', {'op': 'entropy', 'prefix': [1]})))
                except TraceDeadline:
                    results.append((name, 'timeout'))
            a = threading.Thread(target=call, args=(.3, 'a')); a.start()
            until = time.monotonic() + 2
            while not list((base / 'requests').glob('*.json')) and time.monotonic() < until:
                time.sleep(.005)
            request = next((base / 'requests').glob('*.json')); stamp = request.stat().st_mtime_ns
            b = threading.Thread(target=call, args=(3, 'b')); b.start()
            a.join(2); self.assertFalse(a.is_alive()); self.assertTrue(request.exists())
            self.assertEqual(request.stat().st_mtime_ns, stamp)
            self.assertEqual(len(claim_batch(base, 8)), 1)
            publish(base, request, {'entropy': .7})
            b.join(2); self.assertFalse(b.is_alive())
            self.assertIn(('a', 'timeout'), results)
            self.assertEqual(dict(results)['b']['entropy'], .7)
            self.assertEqual(list((base / 'leases').glob('*/*.json')), [])

    def test_last_waiter_timeout_removes_orphan(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(TraceDeadline):
                RPC(root, Budget(root / 'budget.json', .03)).call('teacher', {'op': 'generate'})
            self.assertEqual(list((root / 'teacher/requests').glob('*.json')), [])

    def test_manifest_migration_uses_exact_json_representation(self):
        import hashlib
        digest = lambda x: hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = {'settings': {'random_rates': []}, 'models': {r: {'model': r, 'protocol': 1, 'worker_sha256': 'old'} for r in ('student', 'teacher')}, 'code_sha256': {'a': 'old'}}
            target = json.loads(json.dumps(source)); target['settings']['random_rates'] = ()
            for m in target['models'].values():
                m.update(protocol=2, worker_sha256='new')
            target['code_sha256'] = {'a': 'new'}
            atomic(root / 'math/manifest.json', source)
            atomic(root / 'recovery_104008_performance/migration.json', {'source_manifests': {'math': source}, 'target_code_sha256': target['code_sha256']})
            migrate_manifest(root / 'math/manifest.json', target, atomic, digest)
            self.assertEqual(target, json.loads((root / 'math/manifest.json').read_text()))
            migrate_manifest(root / 'math/manifest.json', target, atomic, digest)
            target['settings']['random_rates'] = [1]
            with self.assertRaises(ValueError):
                migrate_manifest(root / 'math/manifest.json', target, atomic, digest)

    def test_protocol_migration_rejects_scientific_changes(self):
        source = {'settings': {'threshold': 2}, 'models': {r: {'model': r, 'protocol': 1, 'worker_sha256': 'old'} for r in ('student', 'teacher')}}
        target = json.loads(json.dumps(source))
        for m in target['models'].values():
            m.update(protocol=2, worker_sha256='new')
        compatible(source, target)
        target['settings']['threshold'] = 3
        with self.assertRaises(ValueError):
            compatible(source, target)


if __name__ == '__main__':
    unittest.main()
