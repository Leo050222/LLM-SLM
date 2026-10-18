import copy
import unittest
from online_rollout.core import Settings
from online_rollout.random_policy import random_rollout,sample_positions
class Backend:
    max_model_len=40960
    def __init__(self,eos=False):self.eos=eos
    def generate(self,role,prefix,n,seed):
        return dict(token_ids=[2 if role=='teacher' else 1]*(min(3,n) if self.eos else n),finish_reason='stop' if self.eos else 'length')
class Tests(unittest.TestCase):
    def test_nonoverlap(self):
        for length in [0,1,8,63,64,1200]:
            p=sample_positions(length,8,8,12)
            self.assertEqual(p,sample_positions(length,8,8,12))
            self.assertLessEqual(len(p),8)
            self.assertTrue(all(b-a>=8 for a,b in zip(p,p[1:])))
            self.assertTrue(all(0<=a and a+8<=length for a in p))
    def test_eight_takeovers_and_resume(self):
        row=dict(dataset_index=0,prompt_token_ids=[9],random_positions=[0,8,32,75,99,130,190,260])
        states=[];s=Settings(max_tokens=400)
        full=random_rollout(row,s,Backend(),save=lambda x:states.append(copy.deepcopy(x)))
        bridges=[e for e in full['events'] if 'bridge_token_ids' in e]
        self.assertEqual([e['anchor_response_position'] for e in bridges],row['random_positions'])
        self.assertEqual(full['token_sources'].count('teacher'),64)
        self.assertEqual(full['judge_calls'],0)
        self.assertEqual(random_rollout(row,s,Backend(),states[2]),full)
    def test_early_eos(self):
        row=dict(dataset_index=0,prompt_token_ids=[9],random_positions=[30])
        r=random_rollout(row,Settings(),Backend(True))
        self.assertEqual(r['token_sources'],['student']*3)
    def test_teacher_eos(self):
        row=dict(dataset_index=0,prompt_token_ids=[9],random_positions=[0])
        r=random_rollout(row,Settings(),Backend(True))
        self.assertEqual(r['token_sources'],['teacher']*3)
if __name__=='__main__':unittest.main()
