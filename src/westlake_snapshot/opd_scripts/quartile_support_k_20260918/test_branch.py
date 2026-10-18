import unittest,random,copy
from branch import *
from prepare import position
class Test(unittest.TestCase):
 def row(self):
  return dict(prompt_token_ids=[10],anchor=dict(response_prefix_token_ids=[20,21]))
 def test_shared_bridge_and_greedy(self):
  row=self.row();calls=[]
  def call(role,prefix,n,temp,r):
   calls.append((role,prefix[:],n,temp))
   return dict(token_ids=list(range(100,132)),finish_reason='length') if role=='teacher' else dict(token_ids=[99],finish_reason='stop')
  b=make_bridge(row,call)
  for k in KS:
   s=run_branch(row,k,b,call);verify(row,k,b,s)
   self.assertEqual(calls[-1][1],[10,20,21]+list(range(100,100+k)))
   self.assertEqual(calls[-1][3],0.)
  self.assertEqual(sum(c[0]=='teacher' for c in calls),1);self.assertEqual(calls[0][2],32)
 def test_early_eos(self):
  row=self.row();b=make_bridge(row,lambda *a:dict(token_ids=[4,5,6],finish_reason='stop'))
  for k in (4,8,16,32):
   s=run_branch(row,k,b,lambda *a:self.fail('No continuation after teacher EOS'));verify(row,k,b,s)
  s=run_branch(row,1,b,lambda *a:dict(token_ids=[7],finish_reason='stop'));self.assertEqual(s['response_token_ids'],[20,21,4,7])
 def test_resume(self):
  row=self.row();b=make_bridge(row,lambda *a:dict(token_ids=list(range(32)),finish_reason='length'));saved=[]
  def save(s):
   saved.append(copy.deepcopy(s))
   if s['student_rounds']==1:raise RuntimeError('interrupt')
  with self.assertRaises(RuntimeError):run_branch(row,4,b,lambda *a:dict(token_ids=[77],finish_reason='length'),save=save)
  calls=[]
  def call(*a):calls.append(a);return dict(token_ids=[88],finish_reason='stop')
  s=run_branch(row,4,b,call,saved[-1]);self.assertEqual(calls[0][1], [10,20,21,0,1,2,3,77]);verify(row,4,b,s)
  self.assertIs(run_branch(row,4,b,lambda *a:self.fail(),s),s)
 def test_quartiles(self):
  for length in [4,5,101,40264]:
   for q in range(4):
    a=position(length,q,random.Random(11));self.assertTrue(q*length//4<=a<(q+1)*length//4)
    self.assertEqual(a,position(length,q,random.Random(11)))
 def test_mismatch(self):
  row=self.row();b=make_bridge(row,lambda *a:dict(token_ids=list(range(32)),finish_reason='length'))
  s=run_branch(row,4,b,lambda *a:dict(token_ids=[7],finish_reason='stop'))
  with self.assertRaises(AssertionError):run_branch(row,8,b,lambda *a:None,s)
if __name__=='__main__':unittest.main()
