import os,json,time,fcntl,subprocess
from pathlib import Path
from ablation_core import atomic_json
P=Path(os.environ['GRID_ROOT']);Q=Path(os.environ['OPD_ONLINE_RUN'])/'queue';S=Path(__file__).parent
configs=json.loads((P/'configs.json').read_text())
for _ in range(720):
 if len(list(Q.glob('replica_*/*/model.json')))==8:break
 time.sleep(5)
else:raise RuntimeError('Models not ready')
(P/'claims').mkdir(exist_ok=True)
for c in configs:
 if (P/c['name']/'VERIFIED.json').exists():continue
 lock=(P/'claims'/(c['name']+'.lock')).open('a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:lock.close();continue
 try:
  if (P/c['name']/'VERIFIED.json').exists():continue
  atomic_json(P/'claims'/(c['name']+'.json'),dict(slot=os.environ['GRID_SLOT'],job=os.environ.get('SLURM_JOB_ID'),started=time.time()))
  cmd=[os.environ['OPD_PYTHON'],str(S/'run.py'),'--mode','nojudge_grid','--input',str(P/'prompts.jsonl'),'--output-dir',str(P/c['name']),'--queue',str(Q),'--replicas','4','--tokenizer','/storage/yukaichengLab/share/model/Qwen3-1.7B','--judge-config',str(S/'unused_judge.json'),'--expected-rows','2254','--concurrency','128','--chunk-tokens','64','--teacher-tokens',str(c['k']),'--gap-threshold',str(c['threshold']),'--max-tokens','40264','--seed','20260911']
  print('START',c['name'],flush=True)
  with (Path(os.environ['OPD_ONLINE_RUN'])/'logs'/f"{c['name']}.log").open('ab') as log:
   for attempt in range(3):
    if subprocess.run(cmd,stdout=log,stderr=log).returncode==0:break
   else:raise RuntimeError('Generation failed: '+c['name'])
  subprocess.run([os.environ['OPD_PYTHON'],str(S/'verify.py'),c['name']],check=True)
  print('DONE',c['name'],flush=True)
 finally:lock.close()
if all((P/c['name']/'VERIFIED.json').exists() for c in configs):
 subprocess.run([os.environ['OPD_PYTHON'],str(S/'verify.py'),'all'],check=True)
atomic_json(Path(os.environ['OPD_ONLINE_RUN'])/'ALL_GENERATION_COMPLETE.json',dict(slot_done=True))
