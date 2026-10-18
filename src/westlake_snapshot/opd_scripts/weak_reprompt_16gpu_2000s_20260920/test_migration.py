import unittest,tempfile,copy,json
from pathlib import Path
from migration import migrate
from deadline import Budget,TraceDeadline
class TestMigration(unittest.TestCase):
 def test_only_resources_and_deadline(self):
  old=dict(execution=dict(replicas=4,concurrency=128),runtime_policy=dict(trace_limit_seconds=900,replicas=4,concurrency=128,code_sha256={}),settings=dict(seed=123),models=dict(weak='same'),input_sha256='same')
  new=copy.deepcopy(old);new['execution']=dict(replicas=8,concurrency=256);new['runtime_policy'].update(trace_limit_seconds=2000,replicas=8,concurrency=256,code_sha256={'test':'new'})
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'manifest.json';p.write_text(json.dumps(old));bad=copy.deepcopy(new);bad['settings']['seed']=124
   with self.assertRaises(AssertionError):migrate(p,bad)
   self.assertEqual(json.loads(p.read_text()),old)
   migrate(p,new);migrate(p,new);self.assertEqual(json.loads(p.read_text()),new)
 def test_cumulative_budget(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'time.json';p.write_text(json.dumps(dict(elapsed_seconds=901)))
   now=[0];b=Budget(p,limit=2000,clock=lambda:now[0]);b.check();now[0]=1098;b.check();now[0]=1099
   with self.assertRaises(TraceDeadline):b.check()
if __name__=='__main__':unittest.main()
