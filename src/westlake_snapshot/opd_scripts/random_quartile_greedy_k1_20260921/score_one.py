"""Run inside a network/filesystem isolated bubblewrap process only."""
import ast, contextlib, io, json, sys
from pathlib import Path
sys.path[:0]=['/repos/evalplus','/repos/LiveCodeBench','/repos/cruxeval/evaluation']
def lcb_child(sample, solution, conn):
 from lcb_runner.evaluation.testing_util import run_test
 try:
  results,meta=run_test(sample,test=solution,debug=False,timeout=6)
  ok=bool(results) and all(v==True for v in results)
  conn.send(dict(correct=bool(ok),status='pass' if ok else 'fail',test_results=[int(v) for v in results]))
 except BaseException as e:
  conn.send(dict(harness_error=type(e).__name__+': '+str(e)))
 finally:conn.close()

def score(x):
 ds=x['dataset'];p=x['problem'];solution=x['solution']
 if ds.startswith('cruxeval'):
  from utils_general import evaluate_score
  mode='input' if ds.endswith('_i') else 'output'
  g='f('+solution+')' if mode=='input' else solution
  ok=evaluate_score(([g],(p['code'],p['input'],p['output']),mode))[0]
  return dict(correct=bool(ok),status='pass' if ok else 'fail')
 if ds in ['humaneval_plus','mbpp_plus']:
  from evalplus.eval import untrusted_check
  from evalplus.gen.util import trusted_exec
  from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
  dataset='mbpp' if ds=='mbpp_plus' else 'humaneval'
  if dataset=='mbpp':
   tree=ast.parse(Path('/repos/evalplus/evalplus/data/mbpp.py').read_text())
   func=next(f for f in tree.body if isinstance(f,ast.FunctionDef) and f.name=='mbpp_deserialize_inputs')
   ns={};exec(compile(ast.Module(body=[func],type_ignores=[]),'official_mbpp_deserialize','exec'),ns)
   for split in ['base','plus']:p[split+'_input']=ns[func.name](p['task_id'],p[split+'_input'])
  status={}
  for split in ['base','plus']:
   inputs=p[split+'_input']
   expected, times=trusted_exec(p['prompt']+p['canonical_solution'],inputs,p['entry_point'],record_time=True,output_not_none=dataset=='mbpp' and p['entry_point'] in MBPP_OUTPUT_NOT_NONE_TASKS)
   st, details=untrusted_check(dataset,solution,inputs,p['entry_point'],expected=expected,atol=p['atol'],ref_time=times,fast_check=True)
   status[split]=st
  return dict(correct=all(v=='pass' for v in status.values()),status=status)
 if ds=='livecodebench':
  from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
  from lcb_runner.evaluation.testing_util import run_test
  sample=CodeGenerationProblem(**p).get_evaluation_sample()
  import multiprocessing
  ctx=multiprocessing.get_context('fork');recv,send=ctx.Pipe(duplex=False)
  child=ctx.Process(target=lcb_child,args=(sample,solution,send));child.start();send.close()
  # Keep execution failures within the per-solution child, as the official runner does.
  limit=min(890,7*len(json.loads(sample['input_output'])['inputs'])+5)
  if recv.poll(limit):
   try:result=recv.recv()
   except EOFError:result=None
  else:result=None
  child.join(1)
  if child.is_alive():
   child.kill();child.join();return dict(correct=False,status='scoring_execution_timeout',execution_limit_seconds=limit)
  if result is None:
   if child.exitcode != 0:return dict(correct=False,status='solution_process_crash',exitcode=child.exitcode)
   raise RuntimeError('LCB child exited without result')
  if 'harness_error' in result:raise RuntimeError(result['harness_error'])
  return result
 raise ValueError(ds)
if __name__=='__main__':
 x=json.load(sys.stdin)
 with contextlib.redirect_stdout(io.StringIO()):result=score(x)
 print('OPD_SCORE_RESULT='+json.dumps(result),file=sys.__stdout__,flush=True)
