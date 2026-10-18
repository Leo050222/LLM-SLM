import os,sys,json,time,signal,subprocess,hashlib
from pathlib import Path
from ablation_core import atomic_json
P=Path(os.environ['BENCH_ROOT']);S=Path(__file__).parent;PY=sys.executable

def stop(proc):
 if proc.poll() is None:
  try:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=30)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()

def main():
 devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(devices)==2
 out=P/'runs'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True,exist_ok=True)
 atomic_json(out/'MANIFEST.json',dict(devices=devices,job=os.environ['SLURM_JOB_ID'],code={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in S.glob('*.py')},phases=['split8','split4','colocated4','split8_repeat'],note='same54workload keys; initialization/warmup excluded; no MPS, no quantization, no scientific wall-time censor'))
 results={}
 for phase in ['split8','split4','colocated4','split8_repeat']:
  slot=out/phase;slot.mkdir(exist_ok=True);procs=[];logs=[];monitor=None;bench=None;started=time.time()
  coloc=phase=='colocated4';pairs=2 if coloc else 1;batch=4 if phase in ['split4','colocated4'] else 8
  try:
   for i in range(pairs):
    for role in ['teacher','student']:
     gpu=devices[i] if coloc else devices[0 if role=='student' else 1]
     key=f'{i}_{role}';env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',VLLM_WORKER_MULTIPROC_METHOD='spawn',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',VLLM_PORT=str(22000+i*2048+(0 if role=='teacher' else 512)),XDG_CACHE_HOME=str(slot/'cache'/key/'xdg'),TORCHINDUCTOR_CACHE_DIR=str(slot/'cache'/key/'torch'),TRITON_CACHE_DIR=str(slot/'cache'/key/'triton'))
     model='/storage/yukaichengLab/share/model/'+('Qwen3-14B' if role=='teacher' else 'Qwen3-1.7B')
     cmd=[PY,str(S/'worker.py'),'--queue',str(slot/f'queue/replica_{i}'),'--role',role,'--model',model,'--batch-size',str(batch)]
     if coloc:cmd+=['--kv-cache-gib','18' if role=='teacher' else '10','--gpu-memory-utilization','.65' if role=='teacher' else '.23']
     log=(slot/f'{key}.log').open('w');logs.append(log);proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True);procs.append(proc)
     ready=slot/f'queue/replica_{i}'/role/'ready.json';deadline=time.monotonic()+1200
     while not ready.exists():
      if proc.poll() is not None:raise RuntimeError(f'{phase}/{key} worker exited {proc.returncode}')
      if time.monotonic()>deadline:raise RuntimeError(f'{phase}/{key} startup timeout')
      time.sleep(2)
   atomic_json(slot/'STARTUP.json',dict(seconds=time.time()-started,pairs=pairs,batch=batch))
   gpu_log=(slot/'gpu.csv').open('w');logs.append(gpu_log)
   monitor=subprocess.Popen(['nvidia-smi','--id='+','.join(devices),'--query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw','--format=csv,nounits','-l','2'],stdout=gpu_log,stderr=subprocess.DEVNULL,start_new_session=True)
   log=(slot/'bench.log').open('w');logs.append(log)
   bench=subprocess.Popen([PY,str(S/'bench.py')],env=dict(os.environ,BENCH_SLOT=str(slot),BENCH_PAIRS=str(pairs)),stdout=log,stderr=log,start_new_session=True)
   while bench.poll() is None:
    if any(p.poll() is not None for p in procs):raise RuntimeError('Worker died during benchmark')
    time.sleep(2)
   if bench.returncode:raise RuntimeError(f'Benchmark exited {bench.returncode}')
   results[phase]=json.loads((slot/'SUMMARY.json').read_text());print('PHASE_COMPLETE',phase,results[phase]['rollouts_per_hour'],flush=True)
  except Exception as e:
   results[phase]=dict(error=type(e).__name__,message=str(e));atomic_json(slot/'PHASE_FAILED.json',results[phase]);print('PHASE_FAILED',phase,str(e),flush=True)
  finally:
   if bench:stop(bench)
   if monitor:stop(monitor)
   for i in range(pairs):
    q=slot/f'queue/replica_{i}';q.mkdir(parents=True,exist_ok=True);(q/'STOP').touch()
   for proc in procs:
    try:proc.wait(timeout=40)
    except subprocess.TimeoutExpired:stop(proc)
   for log in logs:log.close()
  # Workers are shut down before the next phase; never evict other jobs.
  time.sleep(5)
 atomic_json(out/'RESULTS.json',results)
 if any('error' in v for v in results.values()):raise RuntimeError('At least one phase failed; inspect per-phase logs')
 base=results['split8']['rollouts_per_hour'];col=results['colocated4']['rollouts_per_hour']
 atomic_json(out/'COMPARISON.json',dict(colocated_vs_current=col/base,colocated_vs_split_batch4=col/results['split4']['rollouts_per_hour'],baseline_repeat_ratio=results['split8_repeat']['rollouts_per_hour']/base,metrics=results,caveats=['single selected workload; not full-benchmark result','numerical/batching differences may change rollout lengths; examine tokens/sec too','GPU utilization is sampled, not a kernel occupancy metric']))
 (out/'BENCHMARK_COMPLETE').touch()
if __name__=='__main__':main()
