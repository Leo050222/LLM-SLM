import copy,unittest
from generation import generate,verify
MODEL={'max_model_len':40960,'eos_token_id':99}
ROW={'dataset_index':1,'prompt_token_ids':[1,2],'anchor':{'response_prefix_token_ids':[77,88]}}
class RPC:
 def __init__(self):self.calls=[]
 def call(self,role,p):
  self.calls.append((role,p));assert role=='teacher' and 'banned_first' not in p
  if len(self.calls)==1:return {'token_ids':[3,4],'finish_reason':'length'}
  return {'token_ids':[99],'finish_reason':'stop'}
class Tests(unittest.TestCase):
 def test_original_prompt_and_teacher_only(self):
  rpc=RPC();out=generate(ROW,17,MODEL,rpc);verify(ROW,17,MODEL,out)
  self.assertEqual(rpc.calls[0][1]['prefix'],[1,2]);self.assertEqual(rpc.calls[1][1]['prefix'],[1,2,3,4]);self.assertEqual(out['token_sources'],['teacher']*3)
 def test_resume_and_seed(self):
  saved=[]
  class Halt(Exception):pass
  def save(s):saved.append(copy.deepcopy(s));raise Halt()
  with self.assertRaises(Halt):generate(ROW,17,MODEL,RPC(),save=save)
  rpc=RPC();rpc.calls=[('prior',{})];out=generate(ROW,17,MODEL,rpc,state=saved[0]);verify(ROW,17,MODEL,out)
  full=RPC();generate(ROW,17,MODEL,full)
  self.assertEqual(rpc.calls[1][1],full.calls[1][1])
  with self.assertRaises(AssertionError):generate(ROW,18,MODEL,RPC(),state=saved[0])
 def test_context_limit(self):
  m=dict(MODEL,max_model_len=5);rpc=RPC();out=generate(ROW,17,m,rpc);verify(ROW,17,m,out);self.assertEqual(out['finish_reason'],'context_limit')
if __name__=='__main__':unittest.main()
