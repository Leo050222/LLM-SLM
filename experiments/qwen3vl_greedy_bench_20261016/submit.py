#!/usr/bin/env python3
"""Submit one startup gate, then three bounded full-inference Slurm allocations."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

CODE = Path(__file__).resolve().parent
RUN = Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_vlm/qwen3vl_instruct_greedy_20261016')
PLAN = [('Qwen3-VL-2B-Instruct', 1), ('Qwen3-VL-4B-Instruct', 1), ('Qwen3-VL-8B-Instruct', 2)]


def save(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(tmp,path)


def sbatch(*args):
    result=subprocess.run(['/soft/slurm/bin/sbatch','--parsable',*map(str,args)],
                          check=True,capture_output=True,text=True)
    job=result.stdout.strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError('Unexpected sbatch response: '+result.stdout)
    return job


def main():
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=RUN/'SUBMISSION.json'
        if path.exists():
            raise RuntimeError('Existing SUBMISSION.json; inspect existing jobs before any retry')
        assert sum(nodes for _,nodes in PLAN)==4
        prepared=json.loads((RUN/'prepared/PREPARED.json').read_text())
        tasks_hash=hashlib.sha256((RUN/'prepared/tasks.jsonl').read_bytes()).hexdigest()
        assert tasks_hash==prepared['tasks_sha256'] and prepared['tasks']==9344
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in CODE.iterdir()
               if p.is_file() and p.suffix in ('.py','.slurm')}
        state={'state':'SUBMITTING','created_at':time.time(),'tasks_sha256':tasks_hash,
               'code_sha256':files,'models':[], 'max_simultaneous_nodes':4,
               'max_simultaneous_gpus':32,'tasks_per_model':9344,'expected_generations':28032,
               'partition':'yukaichenglab','account':'yukaicheng','temperature':0,
               'max_model_len':262144,'max_tokens_policy':'262144 - expanded prompt tokens',
               'smoke_job_id':None}
        save(path,state)
        try:
            smoke=sbatch(CODE/'smoke.slurm')
            state['smoke_job_id']=smoke
            save(path,state)
            for model,nodes in PLAN:
                job=sbatch('--job-name=vlm_'+model.split('-')[2], '--nodes='+str(nodes),
                           '--ntasks='+str(nodes), '--dependency=afterok:'+smoke,
                           CODE/'inference.slurm',model,nodes*8,smoke)
                state['models'].append({'model':model,'nodes':nodes,'gpus':nodes*8,
                                        'shards':nodes*8,'job_id':job,'depends_on':smoke})
                save(path,state)
            state['state']='SUBMITTED'
            save(path,state)
        except BaseException as error:
            state.update(state='PARTIAL_SUBMISSION',error=str(error))
            save(path,state)
            raise
        print(json.dumps(state,indent=2))


if __name__=='__main__':
    main()
