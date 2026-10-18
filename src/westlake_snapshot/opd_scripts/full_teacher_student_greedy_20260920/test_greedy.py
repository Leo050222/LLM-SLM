import ast, unittest
from pathlib import Path
from ablation_core import Settings
from generation import generate
from test_generation import RPC,row,model
class GreedyTests(unittest.TestCase):
 def test_reject_sampling_settings(self):
  with self.assertRaises(AssertionError):generate(row,Settings(mode='student',temperature=1.),model,RPC())
 def test_worker_generate_is_greedy(self):
  tree=ast.parse(Path(__file__).with_name('worker.py').read_text())
  calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='SamplingParams']
  generation=[n for n in calls if any(k.arg=='seed' and isinstance(k.value,ast.Subscript) for k in n.keywords)]
  self.assertEqual(len(generation),1)
  kw={k.arg:k.value for k in generation[0].keywords if k.arg}
  self.assertEqual(ast.literal_eval(kw['temperature']),0.)
  self.assertEqual(ast.literal_eval(kw['top_p']),1.)
  self.assertEqual(ast.literal_eval(kw['top_k']),-1)
if __name__=='__main__':unittest.main()
