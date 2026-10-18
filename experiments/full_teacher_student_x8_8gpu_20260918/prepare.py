from pathlib import Path
import json,hashlib,shutil
from dataclasses import asdict
from ablation_core import Settings,atomic_json,digest
from generation import verify
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');P=R/'opd_eval_baselines/full_teacher_student_x8_20260918'
def main():
 P.mkdir(parents=True,exist_ok=True)
 configs=[];inputmeta={};reused=0
 for family,rel,count in [('code','opd_eval_code/code_grid_x8_20260916',2317),('math','opd_eval_hard/math_reasoning_grid_x8_20260916',2254)]:
  src=R/rel;raw=(src/'prompts.jsonl').read_bytes();rows=list(map(json.loads,raw.splitlines()));assert len(rows)==count
  dest=P/f'{family}_prompts.jsonl'
  if dest.exists():assert dest.read_bytes()==raw
  else:dest.write_bytes(raw)
  inputmeta[family]=dict(rows=count,sha256=hashlib.sha256(raw).hexdigest(),source=str(src))
  for role in ['student','teacher']:
   old=R/'opd_eval_code/code_reasoning_grid_20260913'/f'{role}_only'
   oldmanifest=json.loads((old/'manifest.json').read_text())
   base=oldmanifest['settings'] if family=='code' else json.loads((src/'base_settings.json').read_text())
   for rep in range(8):
    settings=dict(base,mode=role,seed=20260911+rep);name=f'{family}_{role}_r{rep}'
    c=dict(name=name,family=family,role=role,replicate=rep,seed=20260911+rep,settings=settings);configs.append(c)
    d=P/name;d.mkdir(exist_ok=True)
    if family=='code' and rep==0:
     assert oldmanifest['settings']==settings and oldmanifest['input_sha256']==digest(rows)
     assert oldmanifest['sampling']==dict(temperature=1.,top_p=1.) and oldmanifest['baseline_generation'].startswith('unconstrained 64-token chunks')
     assert oldmanifest['runtime_policy']['trace_limit_seconds']==900
     assert oldmanifest['models'][role]['model'].endswith('Qwen3-1.7B' if role=='student' else 'Qwen3-14B')
     assert len(list((old/'completed').glob('*.json')))==count
     (d/'completed').mkdir(exist_ok=True)
     for row in rows:
      f=old/'completed'/f"{row['dataset_index']:05d}.json";x=json.loads(f.read_text());cfg=Settings(**settings);verify(row,cfg,x)
      assert x['dataset']==row['dataset'];dst=d/'completed'/f.name
      if dst.exists():assert dst.read_bytes()==f.read_bytes()
      else:shutil.copyfile(f,dst)
      reused+=1
     atomic_json(d/'REUSED.json',dict(source=str(old),rows=count,source_manifest=oldmanifest,note='generation reused byte-for-byte; transport/deployment differs; scores recomputed'))
 meta=dict(inputs=inputmeta,seeds=list(range(20260911,20260919)),roles=['student','teacher'],expected_outputs=73136,new_outputs=68502,reused_outputs=reused,generation_start='original_prompt',trace_limit_seconds=900)
 assert reused==4634
 for name,value in [('configs.json',configs),('INPUTS_COMPLETE.json',meta)]:
  f=P/name
  if f.exists():assert json.loads(f.read_text())==value
  else:atomic_json(f,value)
 (P/'judge').mkdir(exist_ok=True);shutil.copyfile(Path(__file__).parent/'system_prompt.txt',P/'judge/system_prompt.txt')
 print('PREPARED',json.dumps(meta),flush=True)
if __name__=='__main__':main()
