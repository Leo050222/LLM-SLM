import copy
import json
import tempfile
import unittest
from pathlib import Path
from migration import migrate

class ManifestTests(unittest.TestCase):
    def test_initialize_resume_and_reject_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'group/manifest.json'
            m=dict(execution=dict(replicas=3,concurrency=96),runtime_policy=dict(trace_limit_seconds=900),settings=dict(mode='random_quartile',teacher_tokens=1,seed=20260912),input_sha256='frozen')
            migrate(p,m)
            self.assertEqual(json.loads(p.read_text()),m)
            migrate(p,copy.deepcopy(m))
            changed=copy.deepcopy(m);changed['settings']['seed']+=1
            with self.assertRaises(AssertionError):migrate(p,changed)
            changed=copy.deepcopy(m);changed['input_sha256']='different'
            with self.assertRaises(AssertionError):migrate(p,changed)
    def test_orphan_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'group/manifest.json';(p.parent/'traces').mkdir(parents=True);(p.parent/'traces/00000.json').write_text('{}')
            m=dict(execution=dict(replicas=3,concurrency=96),runtime_policy=dict(trace_limit_seconds=900),settings=dict(mode='random_quartile',teacher_tokens=1))
            with self.assertRaises(AssertionError):migrate(p,m)
if __name__=='__main__':unittest.main()
