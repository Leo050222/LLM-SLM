import json,os,secrets,socket,subprocess,time,shutil
from pathlib import Path
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');P=R/'opd_eval_ablation/main_greedy_k1_t2_20260921';S=R/'opd_scripts/main_greedy_score_20260921';J=P/'judge';PY='/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python'
assert (P/'INPUTS_COMPLETE.json').exists()
assert not (P/'ALL_SCORING_COMPLETE.json').exists()
assert not (S/'LAUNCH.json').exists(),'Already submitted; inspect existing launch'
assert socket.gethostbyname(socket.gethostname())=='192.168.180.240',socket.gethostname()
J.mkdir(exist_ok=True);shutil.copy2(S/'system_prompt.txt',J/'system_prompt.txt')
(J/'EXTERNAL_SCORING_APPROVED.json').write_text(json.dumps(dict(authorization='User explicitly requested: 提交评分',rows=2254,destination='api.openlux.ai',model='gpt-4o-mini',fields=['candidate_answer','reference_answer'],time=time.time()),ensure_ascii=False,indent=2))
f=os.open(J/'.relay_token',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(f,'wb') as stream:stream.write(secrets.token_bytes(32))
with (J/'relay.log').open('ab') as log:
 relay=subprocess.Popen([PY,str(S/'relay.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(20):
 if (J/'relay.json').exists():break
 assert relay.poll() is None,'Relay exited'
 time.sleep(.5)
assert (J/'relay.json').exists()
# Code harness smoke is performed inside the CPU allocation; API probes gate scoring.
job=subprocess.check_output(['/soft/slurm/bin/sbatch','--parsable','--dependency=afterok:108123',str(S/'score.slurm')],text=True).strip()
record=dict(job_id=int(job.split(';')[0]),generation_job=108123,rows=4571,code_rows=2317,math_rows=2254,cpus=32,gpus=0,relay_pid=relay.pid,status='submitted',output_root=str(P),submitted_at=time.time())
(S/'LAUNCH.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
