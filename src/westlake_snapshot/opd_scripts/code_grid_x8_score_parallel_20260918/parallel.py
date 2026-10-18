"""Shard orchestration around the unchanged, fingerprinted scoring implementation."""
import hashlib,os,sys
from pathlib import Path
S=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/code_grid_x8_score_20260918')
sys.path.insert(0,str(S))
source=(S/'score.py').read_text()
EXPECTED='a42a1222b970d27d79590a1753a1810d1d3f0bc370986b16b43639c1bb68d416'
assert hashlib.sha256(source.encode()).hexdigest()==EXPECTED,'Scorer changed; review before resume'
mode=os.environ.get('SCORE_MODE','shard')
def replace(old,new):
 global source
 assert source.count(old)==1,old
 source=source.replace(old,new)
if mode=='shard':
 shard=int(os.environ['SLURM_ARRAY_TASK_ID']);assert 0<=shard<4
 replace("fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)","fcntl.flock(lock,fcntl.LOCK_SH)\n shard_lock=(P/'score_parallel_shard_%d.lock').open('a');fcntl.flock(shard_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)"%shard)
 replace('tasks=queue.Queue()', 'configs=[c for i,c in enumerate(configs) if i%%4==%d]\n tasks=queue.Queue()'%shard)
 replace('assert tasks.qsize()==463400','assert tasks.qsize()==115850')
 replace("P/'SCORING_PROGRESS.json'", "P/'parallel_scoring_20260918/shard_%d_PROGRESS.json'"%shard)
 replace('total=463400','total=115850')
 replace("P/'SCORING_ERRORS.json'", "P/'parallel_scoring_20260918/shard_%d_ERRORS.json'"%shard)
 replace('groups=[]',"atomic(P/'parallel_scoring_20260918/shard_%d_COMPLETE.json',dict(progress,total=115850,completed_at=time.time(),signature=signature))\n print('SHARD_COMPLETE',%d,dict(progress),flush=True)\n return\n groups=[]"%(shard,shard))
elif mode=='aggregate':
 replace('tasks=queue.Queue()',"\n for i in range(4):\n  marker=json.loads((P/f'parallel_scoring_20260918/shard_{i}_COMPLETE.json').read_text())\n  assert marker['signature']==signature and marker['processed']==115850 and marker['errors']==0\n tasks=queue.Queue()")
 replace('for row in rows:tasks.put((c,row))','for row in []:tasks.put((c,row))')
 replace('assert tasks.qsize()==463400','assert tasks.qsize()==0')
else:raise ValueError(mode)
exec(compile(source,str(S/'score.py'),'exec'),{'__file__':str(S/'score.py'),'__name__':'__main__'})
