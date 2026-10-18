from pathlib import Path
import json, hashlib
from dataclasses import asdict
from collections import Counter
from ablation_core import Settings, atomic_json
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo')
P=R/'opd_eval_baselines/full_teacher_student_greedy_20260920'
def main():
 P.mkdir(parents=True,exist_ok=True)
 configs=[]; inputs={}
 for family,rel,count in [('code','opd_eval_code/code_grid_x8_20260916',2317),('math','opd_eval_hard/math_reasoning_grid_x8_20260916',2254)]:
  src=R/rel; raw=(src/'prompts.jsonl').read_bytes();rows=list(map(json.loads,raw.splitlines()))
  assert len(rows)==count and len({r['dataset_index'] for r in rows})==count
  dest=P/f'{family}_prompts.jsonl'
  if dest.exists():assert dest.read_bytes()==raw
  else:dest.write_bytes(raw)
  inputs[family]=dict(rows=count,sha256=hashlib.sha256(raw).hexdigest(),source=str(src),datasets=dict(Counter(r['dataset'] for r in rows)))
  for role in ['student','teacher']:
   source=R/'opd_eval_code/code_reasoning_grid_20260913'/f'{role}_only'/'manifest.json'
   base=json.loads(source.read_text())['settings'] if family=='code' else json.loads((src/'base_settings.json').read_text())
   settings=asdict(Settings(**dict(base,mode=role,seed=20260911,temperature=0.)))
   name=f'{family}_{role}_r0';(P/name).mkdir(exist_ok=True)
   configs.append(dict(name=name,family=family,role=role,replicate=0,seed=20260911,settings=settings))
 meta=dict(inputs=inputs,seeds=[20260911],roles=['student','teacher'],expected_outputs=9142,new_outputs=9142,reused_outputs=0,generation_start='original_prompt',sampling=dict(temperature=0.,top_p=1.,top_k=-1),trace_limit_seconds=900)
 for name,value in [('configs.json',configs),('INPUTS_COMPLETE.json',meta)]:
  f=P/name
  if f.exists():assert json.loads(f.read_text())==value
  else:atomic_json(f,value)
 (P/'judge').mkdir(exist_ok=True)
 (P/'judge/system_prompt.txt').write_bytes((Path(__file__).parent/'system_prompt.txt').read_bytes())
 print('PREPARED',json.dumps(meta),flush=True)
if __name__=='__main__':main()
