import copy,unittest
from dynamic import Settings,rollout,verify_trace,finalize_timeout
from deadline import TraceDeadline

ROW=dict(dataset_index=0,prompt_token_ids=[100],problem='test')

class Backend:
    max_model_len=100
    def __init__(self,entropy=(.5,.2),stop_at=None,fail_entropy=False):
        self.entropy=entropy;self.stop_at=stop_at;self.teacher=0;self.student=0;self.calls=[];self.fail_entropy=fail_entropy
    def generate(self,role,prefix,n,seed,banned_first=None):
        self.calls.append((role,list(prefix),banned_first))
        if role=='student':
            self.student+=1
            return dict(token_ids=[10] if self.student==1 else [99],finish_reason='length' if self.student==1 else 'stop')
        self.teacher+=1
        return dict(token_ids=[20+self.teacher],finish_reason='stop' if self.teacher==self.stop_at else 'length')
    def candidates(self,prefix,ids,threshold):
        return [dict(response_position=0,student_token_id=ids[0],signed_logprob_gap_student_minus_teacher=3)] if self.student==1 else []
    def call(self,role,**req):
        if self.fail_entropy:raise TraceDeadline()
        return dict(entropy=self.entropy[0 if role=='student' else 1],full_vocabulary=True)

class DynamicTest(unittest.TestCase):
    def run_case(self,b):
        x=rollout(ROW,Settings(),b);verify_trace(x);return x
    def test_inclusive_boundary_and_cap8(self):
        b=Backend();x=self.run_case(b)
        self.assertEqual(b.teacher,8);self.assertEqual(len(x['events'][0]['entropy_decisions']),7)
        tc=[c for c in b.calls if c[0]=='teacher'];self.assertEqual(tc[0][2],10)
        self.assertTrue(all(c[2] is None for c in tc[1:]))
    def test_either_condition_false_stops(self):
        for entropy in [(.499,.2),(.5,.201)]:
            b=Backend(entropy);x=self.run_case(b);self.assertEqual(b.teacher,1)
            self.assertEqual(x['events'][0]['bridge_stop_reason'],'entropy_condition_false')
    def test_eos(self):
        b=Backend(stop_at=2);x=self.run_case(b)
        self.assertEqual(b.teacher,2);self.assertEqual(b.student,1);self.assertEqual(x['finish_reason'],'stop')
    def test_timeout_keeps_committed_token(self):
        saved=[];b=Backend(fail_entropy=True)
        with self.assertRaises(TraceDeadline):rollout(ROW,Settings(),b,save=lambda x:saved.append(copy.deepcopy(x)))
        x=finalize_timeout(saved[-1]);verify_trace(x)
        self.assertEqual(x['response_token_ids'],[21]);self.assertEqual(x['finish_reason'],'trace_timeout')
    def test_resume_inside_bridge(self):
        saved=[];b=Backend(fail_entropy=True)
        with self.assertRaises(TraceDeadline):rollout(ROW,Settings(),b,save=lambda x:saved.append(copy.deepcopy(x)))
        b.fail_entropy=False;x=rollout(ROW,Settings(),b,saved[-1]);verify_trace(x)
        self.assertEqual(b.teacher,8);self.assertEqual(x['response_token_ids'][:8],list(range(21,29)))

if __name__=='__main__':unittest.main()
