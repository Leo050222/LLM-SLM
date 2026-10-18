"""One-shot run lifecycle: submit scoring only after the strict generation gate."""
import csv, fcntl, hashlib, json, subprocess, time
from pathlib import Path
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');P=R/'opd_eval_code/code_reasoning_grid_20260913';S=Path(__file__).parent
lock=(P/'finish.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
end=time.monotonic()+14*86400
while not (P/'ALL_GENERATION_COMPLETE.json').exists():
 if time.monotonic()>end:raise TimeoutError('generation gate not reached in 14 days')
 if any(not (f.parent/'RECOVERY_SCHEDULED.json').exists() for f in (P/'slots').glob('*/FAILED')):raise RuntimeError('GPU dispatcher failure: inspect before continuing')
 time.sleep(30)
assert (S/'SCORE_SMOKE_PASS').exists()
m=json.loads((P/'data_manifest.json').read_text())
for name,h in m['sha256'].items():assert hashlib.sha256((P/'data'/name).read_bytes()).hexdigest()==h
submission=P/'score_submission.json'
if not submission.exists():
 job=subprocess.check_output(['/soft/slurm/bin/sbatch','--parsable',str(S/'score.slurm')],text=True).strip()
 submission.write_text(json.dumps(dict(job_id=job,max_nodes=3,cpus_per_node=64)))
 print('SCORE_SUBMITTED',job,flush=True)
while not all((P/f'SCORING_SHARD_{i}_COMPLETE').exists() for i in range(3)):
 if time.monotonic()>end:raise TimeoutError('scoring not complete in 14 days')
 time.sleep(60)
records=[]
for c in json.loads((P/'configs.json').read_text()):
 g=P/c['name'];summary=json.loads((g/'SCORED.json').read_text());stats={ds:dict(interventions=0,teacher=0,tokens=0,timeouts=0) for ds in summary}
 for f in (g/'completed').glob('*.json'):
  x=json.loads(f.read_text());a=stats[x['dataset']]
  a['interventions']+=sum('bridge_token_ids' in e for e in x['events'])
  a['teacher']+=x['token_sources'].count('teacher');a['tokens']+=len(x['token_sources']);a['timeouts']+=x['finish_reason']=='trace_timeout'
 for ds,acc in summary.items():
  a=stats[ds];records.append(dict(config=c['name'],dataset=ds,**acc,mean_interventions=a['interventions']/acc['n'],mean_output_tokens=a['tokens']/acc['n'],teacher_token_fraction=a['teacher']/a['tokens'] if a['tokens'] else 0,timeouts=a['timeouts']))
with (P/'FINAL_COMPARISON.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
(P/'EVALUATION_COMPLETE.json').write_text(json.dumps(dict(rows=62559,summary_rows=len(records),time=time.time())))
print('EVALUATION_COMPLETE',flush=True)
