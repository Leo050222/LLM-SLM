import unittest,copy
from dataclasses import asdict
from ablation_core import Settings,rollout,digest
from reprompt import build_prompt
class Tok:
 def encode(self,s,add_special_tokens=False):return [ord(c)+1000 for c in s]
 def decode(self,ids,skip_special_tokens=False):return ''.join(chr(i) for i in ids)
ROW=dict(problem='Solve x.',prompt_token_ids=[999],dataset_index=0)
S=Settings(mode='weak_reprompt',teacher_tokens=1,max_takeovers=None,max_tokens=5)
class Backend:
 max_model_len=10000
 tokenizer=Tok()
 def __init__(self,gap=3,fail=False,eos=False):self.calls=[];self.gap=gap;self.fail=fail;self.eos=eos
 def generate(self,role,prefix,n,seed,banned_first=None):
  assert role=='student';self.calls.append(('normal',list(prefix)))
  return dict(token_ids=[65,66,67][:n],finish_reason='stop' if len(self.calls)>1 else 'length')
 def candidates(self,prefix,ids,threshold):
  return [dict(response_position=1,student_token_id=ids[1],signed_logprob_gap_student_minus_teacher=self.gap)] if len(ids)>1 and len(self.calls)==1 else []
 def call(self,role,**kw):
  assert role=='student' and kw['n']==1 and kw['banned_first']==66
  self.calls.append(('reprompt',kw['prefix']))
  if self.fail:raise RuntimeError('RPC failure')
  return dict(token_ids=[90],finish_reason='stop' if self.eos else 'length')
class Tests(unittest.TestCase):
 def test_handoff(self):
  b=Backend();x=rollout(ROW,S,b)
  self.assertEqual(x['response_token_ids'][:2],[65,90]);self.assertEqual(b.calls[2],('normal',[999,65,90]))
  self.assertEqual(b.calls[1][1][-1],65);self.assertEqual(x['events'][0]['draft_token_ids'],[65,66,67])
  self.assertEqual(x['token_sources'][:2],['student','student_reprompt'])
 def test_strict(self):
  b=Backend(gap=2);rollout(ROW,S,b);self.assertFalse(any(c[0]=='reprompt' for c in b.calls))
 def test_eos(self):
  b=Backend(eos=True);x=rollout(ROW,S,b);self.assertEqual(len(b.calls),2);self.assertEqual(x['finish_reason'],'stop')
 def test_context(self):
  b=Backend();b.max_model_len=100;x=rollout(ROW,S,b);self.assertEqual(x['finish_reason'],'reprompt_context_limit');self.assertEqual(len(b.calls),1)
 def test_atomic_failure(self):
  state=dict(identity=digest(dict(row=ROW,settings=asdict(S))),response_token_ids=[],token_sources=[],events=[],rounds=0,student_draft_tokens=0,discarded_tokens=0,judge_calls=0,status='RUNNING');before=copy.deepcopy(state)
  with self.assertRaises(RuntimeError):rollout(ROW,S,Backend(fail=True),checkpoint=state)
  self.assertEqual(state,before)
 def test_completed_resume(self):
  b=Backend();x=rollout(ROW,S,b);n=len(b.calls);self.assertEqual(rollout(ROW,S,b,checkpoint=x),x);self.assertEqual(len(b.calls),n)
 def test_prompt(self):
  ids=build_prompt(Tok(),ROW,[65],[66,67,68],1)
  self.assertEqual(ids[-2:],[65,66]);text=''.join(chr(i-1000) for i in ids[:-2]);self.assertIn('Previous attempt:\nABCD',text);self.assertNotIn('<BRANCH_POINT>',text);self.assertTrue(text.endswith('New reasoning:\n'))
 def test_whole_attempt_decoded_once(self):
  class RecordingTok(Tok):
   calls=[]
   def decode(self,ids,skip_special_tokens=False):
    self.calls.append((list(ids),skip_special_tokens))
    return super().decode(ids,skip_special_tokens)
  t=RecordingTok();build_prompt(t,ROW,[65],[66,67,68],1)
  self.assertEqual(t.calls,[([65,66,67,68],True)])
 def test_empty_prefix(self):
  ids=build_prompt(Tok(),ROW,[],[65,66],0)
  text=''.join(chr(i-1000) for i in ids)
  self.assertTrue(text.endswith('New reasoning:\n'))
  self.assertIn('Previous attempt:\nAB',text)
 def test_ban_enforced(self):
  b=Backend();b.call=lambda role,**kw:dict(token_ids=[66],finish_reason='length')
  with self.assertRaises(ValueError):rollout(ROW,S,b)
if __name__=='__main__':unittest.main()
