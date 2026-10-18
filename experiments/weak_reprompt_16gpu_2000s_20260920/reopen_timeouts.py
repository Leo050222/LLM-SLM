"""Run once after old Slurm job stops. Archive timeout completions and reopen checkpoints."""
from pathlib import Path
import json,shutil,fcntl,hashlib
from ablation_core import atomic_json
P=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_ablation/weak_reprompt_t2_k1_20260920')
D=P/'r1_weak_reprompt_t2_k1'
A=P/'recovery_16gpu_2000s'
def main():
 lock=(P/'deadline_scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 A.mkdir(exist_ok=True)
 assert not (A/'REOPEN_COMPLETE.json').exists(),'Already reopened; inspect receipt rather than rerun'
 shutil.copy2(D/'manifest.json',A/'manifest.original.json')
 reopened=[];retained=[]
 # Include checkpoint-only timeout if the old process stopped between checkpoint and completed publication.
 names={f.name for f in (D/'completed').glob('*.json')}|{f.name for f in (D/'traces').glob('*.json')}
 for name in sorted(names):
  completed=D/'completed'/name;trace=D/'traces'/name;timing=D/'timings'/name
  source=completed if completed.exists() else trace
  v=json.loads(source.read_text())
  if v.get('finish_reason')!='trace_timeout':
   if completed.exists():retained.append(dict(name=name,sha256=hashlib.sha256(completed.read_bytes()).hexdigest()))
   continue
  for kind,path in [('completed',completed),('traces',trace),('timings',timing)]:
   if path.exists():
    dest=A/kind/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(path,dest)
  state=json.loads(trace.read_text()) if trace.exists() else v
  assert state['finish_reason']=='trace_timeout'
  elapsed=json.loads(timing.read_text())['elapsed_seconds'];assert 900<=elapsed<2000
  state.update(status='RUNNING',resumed_from_timeout_limit=900)
  for k in ['finish_reason','timeout_count_as_wrong','intervention_audit']:state.pop(k,None)
  atomic_json(trace,state)
  completed.unlink(missing_ok=True)
  reopened.append(dict(name=name,elapsed_seconds=elapsed,response_tokens=len(state['response_token_ids'])))
 for path in [D/'VERIFIED.json',P/'ALL_GENERATION_COMPLETE.json']:
  if path.exists():shutil.move(path,A/path.name)
 atomic_json(A/'REOPEN_COMPLETE.json',dict(previous_job=107636,new_limit_seconds=2000,elapsed_policy='preserved cumulative active time; no reset',reopened=reopened,retained=retained))
 print(json.dumps(dict(reopened=len(reopened),retained=len(retained),pending=4571-len(retained))))
if __name__=='__main__':main()
