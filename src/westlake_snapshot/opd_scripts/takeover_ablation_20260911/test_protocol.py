import unittest
from ablation_core import Settings,rollout
ROW=dict(dataset_index=60,dataset='math500',prompt_token_ids=[99],problem='test')
class Fake:
 max_model_len=10000
 def __init__(self):self.teacher_calls=[];self.score_calls=0
 def generate(self,role,prefix,n,seed,banned_first=None):
  if role=='teacher':self.teacher_calls.append((n,banned_first));token=8 if banned_first is not None else 7
  else:token=7
  return dict(token_ids=[token]*n,finish_reason='length')
 def candidates(self,prefix,ids,threshold):
  self.score_calls+=1
  return [dict(response_position=0,student_token_id=7,absolute_logprob_gap=3,signed_logprob_gap_student_minus_teacher=3)]
def yes(*a):return dict(select_anchor=True)
def forbidden(*a):raise AssertionError('unexpected judge')
class Tests(unittest.TestCase):
 def test_caps_and_nojudge(self):
  for mode in ['luna_cap8','nojudge_cap8']:
   b=Fake();r=rollout(ROW,Settings(max_tokens=200,mode=mode),b,yes if mode.startswith('luna') else forbidden)
   self.assertEqual(len(b.teacher_calls),8);self.assertEqual(b.score_calls,8);self.assertEqual(len(r['response_token_ids']),200)
   self.assertEqual(r['judge_calls'],8 if mode.startswith('luna') else 0)
 def test_unlimited_original_token(self):
  b=Fake();s=Settings(max_tokens=100,mode='luna_allow',max_takeovers=None,first_token_ban=False)
  r=rollout(ROW,s,b,yes);self.assertGreater(len(b.teacher_calls),8);self.assertTrue(all(t==7 for t in r['response_token_ids']))
  self.assertTrue(all(ban is None for _,ban in b.teacher_calls))
 def test_random_no_scoring_nojudge_resume(self):
  b=Fake();s=Settings(max_tokens=100,mode='random_allow',max_takeovers=None,first_token_ban=False,random_rates=(('math500',1.),))
  saved=[]
  r=rollout(ROW,s,b,forbidden,save=lambda v:saved.append(__import__('copy').deepcopy(v)))
  self.assertEqual(b.score_calls,0);self.assertGreater(len(b.teacher_calls),8)
  resumed=rollout(ROW,s,Fake(),forbidden,checkpoint=saved[3]);self.assertEqual(r,resumed)
 def test_negative_and_boundary_rejected(self):
  class Mixed(Fake):
   def candidates(self,prefix,ids,threshold):
    return [dict(response_position=0,student_token_id=7,absolute_logprob_gap=abs(g),signed_logprob_gap_student_minus_teacher=g) for g in [-5.,2.]]
  for mode in ['luna_cap8','nojudge_cap8','luna_allow']:
   b=Mixed();r=rollout(ROW,Settings(max_tokens=100,mode=mode),b,forbidden)
   self.assertFalse(b.teacher_calls);self.assertEqual(r['judge_calls'],0)
 def test_random_zero(self):
  b=Fake();s=Settings(max_tokens=100,mode='random_allow',max_takeovers=None,first_token_ban=False,random_rates=(('math500',0.),))
  r=rollout(ROW,s,b,forbidden);self.assertFalse(b.teacher_calls);self.assertEqual(b.score_calls,0)
if __name__=='__main__':unittest.main()
