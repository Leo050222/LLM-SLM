import unittest
from ablation_core import Settings,rollout
from prepare import plan
class Backend:
    max_model_len=10000
    def __init__(self, eos=1000):self.eos=eos
    def candidates(self,*a):raise AssertionError('Threshold scoring must not run')
    def generate(self,role,prefix,n,seed,banned_first=None):
        pos=len(prefix)-1
        count=min(n,max(1,self.eos-pos))
        return dict(token_ids=[2 if role=='teacher' else 1]*count,finish_reason='stop' if pos+count>=self.eos else 'length')
class Tests(unittest.TestCase):
    def row(self,positions):return dict(dataset_index=3,prompt_token_ids=[9],quartile_plan=dict(random_positions_0based=positions))
    def test_exact(self):
        r=rollout(self.row([0,1,63,64,65,99]),Settings(mode='random_quartile',teacher_tokens=1,max_tokens=130,max_takeovers=None),Backend(),None)
        self.assertEqual([e['anchor_response_position'] for e in r['events'] if 'bridge_token_ids'in e],[0,1,63,64,65,99])
        self.assertTrue(all(t=='student' for t in r['token_sources'][100:]))
    def test_eos(self):
        r=rollout(self.row([3,20,99]),Settings(mode='random_quartile',teacher_tokens=1,max_tokens=130,max_takeovers=None),Backend(10),None)
        self.assertEqual(len(r['response_token_ids']),10);self.assertEqual(r['finish_reason'],'stop')
        self.assertEqual(sum(t=='teacher' for t in r['token_sources']),1)
    def test_partition(self):
        for n in range(101):
            ps=list(range(0,n,2));x=dict(response_token_ids=[1]*n,token_sources=['teacher' if i in ps else 'student' for i in range(n)],events=[dict(anchor_response_position=i,bridge_token_ids=[2]) for i in ps],finish_reason='stop')
            q=plan(x,'test',b'test',dict(dataset='d',problem_id='p',dataset_index=0));ends=q['segment_end_token_counts']
            self.assertEqual([sum(a<=i<b for i in q['random_positions_0based']) for a,b in zip([0]+ends[:-1],ends)],q['intervention_counts'])
            self.assertEqual(q,plan(x,'test',b'test',dict(dataset='d',problem_id='p',dataset_index=0)))
    def test_resume(self):
        settings=Settings(mode='random_quartile',teacher_tokens=1,max_tokens=130,max_takeovers=None);row=self.row([2,4,65]);saved=[]
        import copy
        full=rollout(row,settings,Backend(),None,save=lambda x:saved.append(copy.deepcopy(x)))
        resumed=rollout(row,settings,Backend(),None,checkpoint=saved[0])
        self.assertEqual(full,resumed)
if __name__=='__main__':unittest.main()
