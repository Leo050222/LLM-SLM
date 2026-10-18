import unittest,tempfile,os,json,hashlib,runpy
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
S=Path(__file__).parent
class SubmissionTest(unittest.TestCase):
 def test_gate_and_single_submission(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);report=dict(passed=False,errors=['loop'],results=[{}]*12,code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in S.glob('*.py')})
   (p/'VALIDATION_PASSED.json').write_text(json.dumps(report))
   with patch.dict(os.environ,GRID_ROOT=tmp,SLURM_JOB_ID='123'),patch('subprocess.run',return_value=SimpleNamespace(stdout='456\n')) as submit:
    with self.assertRaises(AssertionError):runpy.run_path(str(S/'submit_formal.py'))
    submit.assert_not_called()
    report.update(passed=True,errors=[]);(p/'VALIDATION_PASSED.json').write_text(json.dumps(report))
    runpy.run_path(str(S/'submit_formal.py'));submit.assert_called_once()
    self.assertIn('--dependency=afterok:123',submit.call_args.args[0])
    self.assertEqual(json.loads((p/'FORMAL_SUBMITTED.json').read_text())['job_id'],456)
    with self.assertRaises(SystemExit):runpy.run_path(str(S/'submit_formal.py'))
    submit.assert_called_once()
if __name__=='__main__':unittest.main()
