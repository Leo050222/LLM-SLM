import json, os, re, subprocess, sys, time, hashlib, concurrent.futures, fcntl
from pathlib import Path
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo')
P=R/'opd_eval_code/code_reasoning_grid_20260913'; S=Path(__file__).parent
ENV=Path('/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd')
def sandbox(payload):
 cmd=['/usr/bin/bwrap','--unshare-all','--die-with-parent','--new-session']
 for src,dest in [(str(ENV),'/env'),('/usr','/usr'),('/lib','/lib'),('/lib64','/lib64'),(str(R/'eval_repos/evalplus'),'/repos/evalplus'),(str(R/'eval_repos/cruxeval/evaluation'),'/repos/cruxeval/evaluation'),(str(R/'eval_repos/LiveCodeBench/lcb_runner'),'/repos/LiveCodeBench/lcb_runner'),(str(S/'score_one.py'),'/score_one.py')]:
  if Path(src).exists():cmd+=['--ro-bind',src,dest]
 cmd+=['--proc','/proc','--dev','/dev','--tmpfs','/tmp','--dir','/work','--chdir','/work','--setenv','HOME','/tmp','--setenv','PATH','/env/bin:/usr/bin','--setenv','OMP_NUM_THREADS','1','--setenv','OPENBLAS_NUM_THREADS','1','--setenv','MKL_NUM_THREADS','1','--setenv','PYTHONDONTWRITEBYTECODE','1','/env/bin/python','/score_one.py']
 # The outer wall cap handles harness-level hangs; inner official test timeouts stay unchanged.
 try:
  out=subprocess.run(cmd,input=json.dumps(payload),text=True,capture_output=True,timeout=900,env={"PATH":"/usr/bin:/bin"})
 except subprocess.TimeoutExpired:
  return dict(correct=False,status='scoring_execution_timeout',execution_limit_seconds=900)
 if out.returncode:raise RuntimeError(out.stderr[-3000:])
 lines=[l[len('OPD_SCORE_RESULT='):] for l in out.stdout.splitlines() if l.startswith('OPD_SCORE_RESULT=')]
 if len(lines)!=1:raise RuntimeError('Expected exactly one structured scorer result')
 return json.loads(lines[0])
def extract(ds,response):
 if ds.startswith('cruxeval'):
  tag='INPUT' if ds.endswith('_i') else 'OUTPUT'
  hits=re.findall(r'\['+tag+r'\](.*?)\[/'+tag+r'\]',response,re.S)
  if hits:return hits[-1].strip()
  return response.strip()
 blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',response,re.S)
 return blocks[0].strip() if blocks else response.strip()
def main():
 selected=os.environ.get('SCORE_GROUPS','').split(',') if os.environ.get('SCORE_GROUPS') else None
 if selected:
  allowed={c['name'] for c in json.loads((P/'configs.json').read_text())}
  assert len(selected)==len(set(selected)) and set(selected)<=allowed
  assert all((P/g/'VERIFIED.json').exists() for g in selected), 'Selected groups must be verified'
 else:
  assert (P/'ALL_GENERATION_COMPLETE.json').exists(),'All rollout groups must finish before scoring'
 shard=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))
 lock=(P/('score_selected_'+hashlib.sha256(','.join(sorted(selected)).encode()).hexdigest()[:12]+'.lock' if selected else f'score_{shard}.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 data={}
 for ds,name,key in [('cruxeval_o','cruxeval.jsonl','id'),('cruxeval_i','cruxeval.jsonl','id'),('humaneval_plus','HumanEvalPlus.jsonl','task_id'),('mbpp_plus','MbppPlus.jsonl','task_id'),('livecodebench','livecodebench_v6_incremental.jsonl','question_id')]:
  items=[json.loads(l) for l in (P/'data'/name).read_text().splitlines()]
  data[ds]={(x['platform']+'/'+x[key] if ds=='livecodebench' else x[key]):x for x in items}
 configs=json.loads((P/'configs.json').read_text())
 configs=[c for c in configs if c['name'] in selected] if selected else configs[shard::3]
 def action(f):
  dest=f.parent.parent/'scores'/f.name
  if dest.exists():
   assert json.loads(dest.read_text())['source_sha256']==hashlib.sha256(f.read_bytes()).hexdigest()
   return
  x=json.loads(f.read_text());ds=x['dataset']
  if x['finish_reason']=='trace_timeout':result=dict(correct=False,status='trace_timeout')
  else:result=sandbox(dict(dataset=ds,problem=data[ds][x['problem_id']],solution=extract(ds,x['response'])))
  result.update(dataset=ds,problem_id=x['problem_id'],dataset_index=x['dataset_index'],source_sha256=hashlib.sha256(f.read_bytes()).hexdigest())
  dest.parent.mkdir(exist_ok=True);tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(result));tmp.replace(dest)
 for c in configs:
  group_lock=(P/c['name']/'scoring.lock').open('a');fcntl.flock(group_lock,fcntl.LOCK_EX)
  files=sorted((P/c['name']/'completed').glob('*.json'));assert len(files)==2317
  with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:list(pool.map(action,files))
  summary={}
  for ds in data:
   scores=[json.loads((P/c['name']/'scores'/f.name).read_text()) for f in files]
   scores=[x for x in scores if x['dataset']==ds]
   summary[ds]=dict(n=len(scores),correct=sum(x['correct'] for x in scores),acc=sum(x['correct'] for x in scores)/len(scores))
  (P/c['name']/'SCORED.json').write_text(json.dumps(summary,indent=2));print(c['name'],summary,flush=True)
  fcntl.flock(group_lock,fcntl.LOCK_UN);group_lock.close()
 (P/('SCORING_SELECTED_'+hashlib.sha256(','.join(sorted(selected)).encode()).hexdigest()[:12]+'_COMPLETE' if selected else f'SCORING_SHARD_{shard}_COMPLETE')).write_text(str(time.time()))
if __name__=='__main__':main()
