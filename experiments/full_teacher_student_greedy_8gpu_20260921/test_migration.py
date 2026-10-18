import unittest,tempfile,json,copy
from pathlib import Path
from migration import migrate
class MigrationTests(unittest.TestCase):
 def test_resume_idempotent_and_reject_changed_seed(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);f=p/'student_manifest.json'
   source=dict(role='student',replicas=2,concurrency=64,trace_limit_seconds=900,seeds=[20260911],models={'model':'same'},code_sha256={'old':'hash'})
   target=dict(source,replicas=8,concurrency=256,code_sha256={'new':'hash'})
   f.write_text(json.dumps(source));(p/'recovery_8gpu_20260921.json').write_text(json.dumps(dict(source_manifests={'student':source},target_code_sha256=target['code_sha256'])))
   migrate(f,target);migrate(f,target);self.assertEqual(json.loads(f.read_text()),target)
   bad=copy.deepcopy(target);bad['seeds']=[9]
   with self.assertRaises(AssertionError):migrate(f,bad)
if __name__=='__main__':unittest.main()
