import copy,unittest
from ablation_core import Settings,digest
from branch import run_branch,verify
class Fake:
 max_model_len=40960
 def __init__(self,eos_teacher=False):self.calls=[];self.scores=0;self.steps=0;self.eos_teacher=eos_teacher
 def generate(self,role,prefix,n,seed,banned_first=None):
  self.calls.append((role,list(prefix),n,seed,banned_first))
  if role=='teacher':return dict(token_ids=[99],finish_reason='stop' if self.eos_teacher else 'length')
  self.steps+=1;return dict(token_ids=[10,11],finish_reason='stop' if self.steps>=6 else 'length')
 def candidates(self,prefix,ids,threshold):
  self.scores+=1;return [dict(response_position=0,student_token_id=10,signed_logprob_gap_student_minus_teacher=3)]
def row():
 return {'dataset_index':0,'prompt_token_ids':[1,2],'anchor':{'response_prefix_token_ids':[3,4],'frozen_prefix_sha256':digest([1,2,3,4]),'entry_token':10,'candidate':dict(response_position=0,student_token_id=10,signed_logprob_gap_student_minus_teacher=3),'source_sha256':'fixed'}}
def settings(cap):return Settings(teacher_tokens=1,gap_threshold=2,mode='nojudge_grid',max_takeovers=cap,seed=17)
class Tests(unittest.TestCase):
 def test_caps_and_no_scoring_after_cap(self):
  seeds=[]
  for cap in range(5):
   b=Fake();out=run_branch(row(),settings(cap),b);self.assertEqual(verify(row(),cap,out),cap)
   self.assertEqual(b.scores,max(0,cap-1));self.assertEqual(sum(c[0]=='teacher' for c in b.calls),cap)
   if cap:seeds.append(b.calls[0][3]);self.assertEqual(b.calls[0][1],[1,2,3,4])
  self.assertEqual(len(set(seeds)),1)
 def test_resume_does_not_repeat_initial_teacher(self):
  saved=[]
  class Halt(Exception):pass
  def save(s):saved.append(copy.deepcopy(s));raise Halt()
  with self.assertRaises(Halt):run_branch(row(),settings(1),Fake(),save=save)
  b=Fake();out=run_branch(row(),settings(1),b,saved[0]);self.assertEqual(verify(row(),1,out),1);self.assertFalse(any(c[0]=='teacher' for c in b.calls));self.assertEqual(b.scores,0)
 def test_teacher_eos(self):
  b=Fake(True);out=run_branch(row(),settings(4),b);self.assertEqual(verify(row(),4,out),1);self.assertEqual(len(b.calls),1)
 def test_input_mismatch(self):
  from branch import initial
  s=initial(row(),settings(1));bad=row();bad['prompt_token_ids']=[7]
  with self.assertRaises(AssertionError):run_branch(bad,settings(1),Fake(),s)
if __name__=='__main__':unittest.main()
