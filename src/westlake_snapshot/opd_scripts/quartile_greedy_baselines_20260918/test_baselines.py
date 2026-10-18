import unittest,copy
from branch import run_branch,verify
from aggregate import summarize,GROUPS
class Test(unittest.TestCase):
 def row(self):return dict(prompt_token_ids=[10],anchor=dict(response_prefix_token_ids=[20,21]))
 def test_roles_same_prefix_greedy(self):
  for role in ['student','teacher']:
   calls=[]
   def call(*a):calls.append(a);return dict(token_ids=[99],finish_reason='stop')
   s=run_branch(self.row(),call,role=role);verify(self.row(),s,role)
   self.assertEqual(calls,[(role,[10,20,21],256,0.,0)])
   self.assertEqual(s['actual_teacher_tokens'],int(role=='teacher'))
 def test_resume_and_role_rejection(self):
  for role in ['student','teacher']:
   saved=[]
   def save(s):saved.append(copy.deepcopy(s));raise RuntimeError('interruption')
   with self.assertRaises(RuntimeError):run_branch(self.row(),lambda *a:dict(token_ids=[77],finish_reason='length'),save=save,role=role)
   calls=[]
   def call(*a):calls.append(a);return dict(token_ids=[88],finish_reason='stop')
   s=run_branch(self.row(),call,saved[-1],role=role);verify(self.row(),s,role);self.assertEqual(calls[0][1],[10,20,21,77])
   self.assertIs(run_branch(self.row(),lambda *a:self.fail(),s,role=role),s)
   with self.assertRaises(AssertionError):run_branch(self.row(),call,s,role='student' if role=='teacher' else 'teacher')
 def test_aggregation(self):
  rs=[dict(dataset='test',group=g,quartile=i//25+1,sample_index=i,correct=(i%2==0 if g=='k0' else True),finish_reason='stop') for g in GROUPS for i in range(100)]
  summary,changes=summarize(rs);self.assertEqual(len(summary),35);self.assertEqual(len(changes),30)
  overall=[r for r in changes if r['quartile']=='all'];self.assertTrue(all(r['accuracy_delta']==.5 and r['wrong_to_correct']==50 for r in overall))
  with self.assertRaises(AssertionError):summarize(rs[:-1])
if __name__=='__main__':unittest.main()
