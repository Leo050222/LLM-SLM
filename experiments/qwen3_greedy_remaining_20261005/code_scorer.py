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
