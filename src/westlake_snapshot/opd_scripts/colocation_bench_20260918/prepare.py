import json,hashlib,random
from pathlib import Path
from ablation_core import atomic_json
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');P=R/'opd_profile/colocation_bench_20260918';source=R/'opd_eval_hard/math_reasoning_grid_x8_20260916'
def main():
 P.mkdir(parents=True,exist_ok=True)
 if (P/'INPUTS_COMPLETE.json').exists():return
 rows=list(map(json.loads,(source/'prompts.jsonl').read_text().splitlines()));picked=[]
 for ds in ['math500','olympiad_bench','bbh']:
  pool=[]
  for r in rows:
   if r['dataset']!=ds:continue
   f=source/'r1_k1_t2/completed'/f"{r['dataset_index']:05d}.json";raw=f.read_bytes();x=json.loads(raw)
   pool.append((len(x['response_token_ids']),r,hashlib.sha256(raw).hexdigest()))
  pool.sort(key=lambda v:(v[0],v[1]['dataset_index']))
  for j,label in enumerate(['short','medium','long']):
   part=pool[len(pool)*j//3:len(pool)*(j+1)//3]
   for length,r,h in random.Random(2026091800+j).sample(part,2):picked.append(dict(r,profile_bin=label,source_length=length,source_sha256=h))
 assert len(picked)==18
 raw=''.join(json.dumps(r)+'\n' for r in picked).encode();(P/'prompts.jsonl').write_bytes(raw)
 atomic_json(P/'INPUTS_COMPLETE.json',dict(rows=18,sha256=hashlib.sha256(raw).hexdigest(),ks=[1,8,32],threshold=2,seed=2026091801,tasks_per_phase=54,selection='two random problems per dataset/source-response-length tercile; not accuracy-selected'))
 print('PREPARED',[(r['dataset'],r['profile_bin'],r['source_length']) for r in picked],flush=True)
if __name__=='__main__':main()
