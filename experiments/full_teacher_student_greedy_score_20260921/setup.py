from pathlib import Path
import fcntl,json,os,secrets,subprocess,sys,time
p=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_baselines/full_teacher_student_greedy_20260920');j=p/'judge';s=Path(__file__).parent
lock=(j/'authorization_setup.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_rescore/grid_math_gpt4o_mini_openlux_20260916/.openlux_api_key').is_file()
auth=dict(authorized_by='user',instruction='跑完自动评分',scope='Both greedy baselines, all eight datasets, 9142 total outputs; up to 4508 math/reasoning grades via existing OpenLux gpt-4o-mini protocol',created_at=time.time())
(j/'EXTERNAL_SCORING_APPROVED.json').write_text(json.dumps(auth,ensure_ascii=False,indent=2))
assert not (j/'relay.json').exists(), 'Relay already configured; inspect rather than duplicate'
f=j/'.relay_token';fd=os.open(f,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'wb') as out:out.write(secrets.token_bytes(32))
log=(j/'relay.log').open('ab')
proc=subprocess.Popen([sys.executable,str(s/'relay.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(30):
 if (j/'relay.json').exists():break
 assert proc.poll() is None, 'Relay exited'
 time.sleep(1)
else:raise RuntimeError('Relay did not start')
port=json.loads((j/'relay.json').read_text())['port']
subprocess.run([sys.executable,str(s/'judge_client.py'),'--tunnel-port',str(port),'--probe'],check=True)
assert not (p/'full_score_job_id.txt').exists()
job=subprocess.check_output(['/soft/slurm/bin/sbatch','--parsable','--dependency=afterany:107946',str(s/'score.slurm')],text=True).strip()
(p/'full_score_job_id.txt').write_text(job+'\n')
print('FULL_SCORE_JOB',job,flush=True)
