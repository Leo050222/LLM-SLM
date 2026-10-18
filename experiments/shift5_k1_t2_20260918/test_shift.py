import copy,unittest
from unittest.mock import patch
from ablation_core import Settings,rollout,draw_shift,digest
from dataclasses import asdict
SET=Settings(mode='shift5',teacher_tokens=1,max_takeovers=None,max_tokens=100)
ROW=dict(dataset='toy',problem_id='p',dataset_index=0,prompt_token_ids=[999])
class Backend:
 max_model_len=1000
 def __init__(self,triggers=(),eos=1000,teacher_stop=False):self.triggers=list(triggers);self.eos=eos;self.teacher_stop=teacher_stop;self.calls=[]
 def generate(self,role,prefix,n,seed,banned_first=None):
  self.calls.append((role,list(prefix),n,banned_first));offset=len(prefix)-1
  if role=='teacher':return dict(token_ids=[777],finish_reason='stop' if self.teacher_stop else 'length')
  count=min(n,max(1,self.eos-offset));return dict(token_ids=list(range(offset+1,offset+count+1)),finish_reason='stop' if offset+count>=self.eos else 'length')
 def candidates(self,prefix,ids,threshold):
  pos=self.triggers.pop(0) if self.triggers else None
  return [] if pos is None else [dict(response_position=pos,student_token_id=ids[pos],signed_logprob_gap_student_minus_teacher=3.)]
class Tests(unittest.TestCase):
 def call(self,b,shift,**kwargs):
  with patch('ablation_core.draw_shift',return_value=shift):return rollout(ROW,SET,b,**kwargs)
 def test_right_cross_chunk(self):
  b=Backend([63],teacher_stop=True);x=self.call(b,5)
  e=x['events'][0];self.assertEqual(e['anchor_response_position'],68);self.assertEqual(e['right_extension_tokens'],5)
  self.assertEqual(b.calls[-1][1],ROW['prompt_token_ids']+list(range(1,69)));self.assertEqual(b.calls[-1][3],69)
 def test_left_cross_chunk(self):
  b=Backend([None,0],teacher_stop=True);x=self.call(b,-5)
  e=x['events'][-1];self.assertEqual(e['anchor_response_position'],59);self.assertEqual(e['rollback_committed_tokens'],5);self.assertEqual(x['response_token_ids'],list(range(1,60))+[777])
 def test_left_prompt_boundary(self):
  x=self.call(Backend([0],teacher_stop=True),-5);e=x['events'][0];self.assertTrue(e['left_boundary_clipped']);self.assertEqual(e['actual_shift'],0);self.assertEqual(x['response_token_ids'],[777])
 def test_previous_teacher_preserved(self):
  b=Backend([0,0],eos=10);x=self.call(b,-5)
  self.assertEqual(x['token_sources'][:2],['teacher','teacher']);self.assertEqual([e['anchor_response_position'] for e in x['events'] if 'bridge_token_ids'in e],[0,1])
 def test_eos_before_target(self):
  b=Backend([7],eos=10);x=self.call(b,5);self.assertEqual(x['finish_reason'],'stop');self.assertEqual(len(x['response_token_ids']),10);self.assertEqual(x['events'][0]['skip_reason'],'eos_before_target');self.assertFalse(any(c[0]=='teacher' for c in b.calls))
 def test_eos_in_extension(self):
  b=Backend([63],eos=66);x=self.call(b,5);self.assertEqual(len(x['response_token_ids']),66);self.assertEqual(x['finish_reason'],'stop');self.assertFalse(any(c[0]=='teacher' for c in b.calls))
 def test_output_limit(self):
  b=Backend([None,35]);x=self.call(b,5);self.assertEqual(len(x['response_token_ids']),100);self.assertEqual(x['events'][-1]['skip_reason'],'length_before_target')
 def test_strict_threshold(self):
  b=Backend(eos=5);b.candidates=lambda prefix,ids,t:[dict(response_position=0,student_token_id=ids[0],signed_logprob_gap_student_minus_teacher=2.)]
  x=self.call(b,1);self.assertEqual(x['token_sources'],['student']*5)
 def test_resume(self):
  saved=[];b=Backend([None,0],teacher_stop=True)
  full=self.call(b,-5,save=lambda s:saved.append(copy.deepcopy(s)))
  resumed=self.call(Backend([0],teacher_stop=True),-5,checkpoint=saved[0]);self.assertEqual(full,resumed)
 def test_failed_teacher_atomic(self):
  state=dict(identity=digest(dict(row=ROW,settings=asdict(SET))),response_token_ids=[],token_sources=[],events=[],rounds=0,student_draft_tokens=0,discarded_tokens=0,judge_calls=0,status='RUNNING');before=copy.deepcopy(state);b=Backend([0]);generate=b.generate
  def fail(role,*a,**kw):
   if role=='teacher':raise TimeoutError()
   return generate(role,*a,**kw)
  b.generate=fail
  with self.assertRaises(TimeoutError):self.call(b,1,checkpoint=state)
  self.assertEqual(state,before)
 def test_shift_reproducible(self):
  a=[draw_shift(ROW,SET,i) for i in range(1000)];self.assertEqual(a,[draw_shift(ROW,SET,i) for i in range(1000)]);self.assertEqual(set(a),{-5,-4,-3,-2,-1,1,2,3,4,5})
if __name__=='__main__':unittest.main()
