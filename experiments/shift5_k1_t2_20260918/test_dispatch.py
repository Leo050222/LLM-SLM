import importlib.util,json,os,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
from dataclasses import asdict
from ablation_core import Settings,atomic_json
class Tokenizer:
 def decode(self,ids,**kwargs):return 'mock response'
class Backend:
 max_model_len=1000
 def __init__(self,*a):pass
 def generate(self,role,prefix,n,seed,**kw):return dict(token_ids=[7] if role=='teacher' else [1,2],finish_reason='stop' if role=='teacher' else 'length')
 def candidates(self,prefix,ids,threshold):return [dict(response_position=0,student_token_id=ids[0],signed_logprob_gap_student_minus_teacher=3.)]
class DispatchTest(unittest.TestCase):
 def test_new_experiment_end_to_end(self):
  fake_run=types.ModuleType('run');fake_run.Backend=Backend;fake_run.tokenizer_fingerprint=lambda t:'test'
  fake_transformers=types.ModuleType('transformers');fake_transformers.AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**kw:Tokenizer())
  with tempfile.TemporaryDirectory() as tmp,patch.dict(sys.modules,{'run':fake_run,'transformers':fake_transformers}):
   root=Path(tmp);slot=root/'slots/job_test';slot.mkdir(parents=True)
   for i in range(4):
    for role in ['student','teacher']:atomic_json(slot/f'queue/replica_{i}/{role}/model.json',dict(model=role,tokenizer_sha256='test',max_model_len=1000))
   row=dict(dataset_index=0,original_dataset_index=3,dataset='toy',problem_id='p',prompt_token_ids=[9])
   (root/'prompts.jsonl').write_text(json.dumps(row)+'\n');atomic_json(root/'configs.json',[dict(name='test',k=1,threshold=2,seed=20260912,replicate_index=1,base_config='shift5_k1_t2')]);atomic_json(root/'base_settings.json',asdict(Settings(mode='shift5',teacher_tokens=1,max_takeovers=None)))
   spec=importlib.util.spec_from_file_location('shift_dispatch',Path(__file__).with_name('dispatch.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
   with patch.dict(os.environ,{'GRID_ROOT':str(root),'OPD_ONLINE_RUN':str(slot),'GRID_REPLICAS':'4','EXPECTED_ROWS':'1'}),patch('ablation_core.draw_shift',return_value=-1):m.main()
   self.assertTrue((root/'ALL_GENERATION_COMPLETE.json').exists());x=json.loads((root/'test/completed/00000.json').read_text());self.assertEqual(x['intervention_audit']['actual_total'],1)
   # Strict same-manifest resume must pass without legacy migration files.
   with patch.dict(os.environ,{'GRID_ROOT':str(root),'OPD_ONLINE_RUN':str(slot),'GRID_REPLICAS':'4','EXPECTED_ROWS':'1'}):m.main()
if __name__=='__main__':unittest.main()
