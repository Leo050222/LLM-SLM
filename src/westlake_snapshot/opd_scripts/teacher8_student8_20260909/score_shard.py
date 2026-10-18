"""Disjoint CPU score shards; reuse the original unmodified scoring function."""
import argparse
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
from pipeline import source,load,sha,atomic,ROOT,H,teacher_path,student_path,validate_teacher,validate_student
from score import score_file

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--rank',type=int,required=True)
    ap.add_argument('--shards',type=int,default=3)
    ap.add_argument('--workers',type=int,default=80)
    args=ap.parse_args();assert 0<=args.rank<args.shards
    out=args.output;cfg=load(out/'config.json')
    assert cfg['limit']==1000
    assert cfg['scorer_sha256']==sha(ROOT/'opd_scripts/run_opd_logit_diagnostic.py')
    assert load(out/'STUDENT_GENERATION_COMPLETE.json')['rows']==256000
    tasks=[];reused=0;total=0
    for r in source(1000):
        i=int(r['dataset_index'])
        if i%args.shards!=args.rank:continue
        tp=teacher_path(out,i);tb=load(tp);validate_teacher(tb,r)
        candidates=[(tp,out/'scores/teacher'/tp.name)]
        for h in H:
            for t in tb['outputs']:
                p=student_path(out,h,i,t['teacher_replicate']);validate_student(load(p),r,t,h)
                candidates.append((p,out/f'scores/h{h}'/p.name))
        for p,dest in candidates:
            total+=1
            if dest.exists():
                b=load(dest);assert b['source_sha256']==sha(p) and len(b['outputs'])==8
                assert b['dataset_index']==i and b['problem_id']==r['problem_id']
                reused+=1
            else:tasks.append((str(p),str(dest)))
    print('SHARD_START',args.rank,'host',os.uname().nodename,'affinity',len(os.sched_getaffinity(0)),'workers',args.workers,'total',total,'reused',reused,'remaining',len(tasks),flush=True)
    start=time.monotonic();done=0;it=iter(tasks)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        pending=set()
        for _ in range(args.workers*2):
            task=next(it,None)
            if task is None:break
            pending.add(pool.submit(score_file,task))
        while pending:
            finished,pending=wait(pending,return_when=FIRST_COMPLETED)
            for future in finished:
                future.result();done+=1
                task=next(it,None)
                if task is not None:pending.add(pool.submit(score_file,task))
            if done%50<len(finished):print('SHARD_PROGRESS',args.rank,done,len(tasks),'seconds',round(time.monotonic()-start,1),flush=True)
    assert done+reused==total
    atomic(out/f'score_shards/rank{args.rank}.json',{'rank':args.rank,'shards':args.shards,'total':total,'reused':reused,'new':done,'job_id':os.environ.get('SLURM_JOB_ID'),'seconds':time.monotonic()-start,'wrapper_sha256':sha(__file__),'scorer_sha256':cfg['scorer_sha256']})
    print('SHARD_COMPLETE',args.rank,total,flush=True)
if __name__=='__main__':main()
