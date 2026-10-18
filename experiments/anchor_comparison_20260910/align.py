import json,ast,re,hashlib
from pathlib import Path
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');E=R/'opd_eval_deepmath';A=E/'ge6_test1000_anchor_bridge_h256_n8x8_seed20260910'
def load(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.open() if x.strip()]
src=R/'ocrd_pipeline/full_rollout/analyze_teacher_rollout_shard.py';tree=ast.parse(src.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='normalized_boolean');exec(compile(ast.Module(body=[fn],type_ignores=[]),str(src),'exec'))
base=lines(E/'ge6_test1000_native_nonthinking_seed20260811/qwen3_1_7b/merged/rollouts.jsonl');base={x['dataset_index']:x for x in base}
m=load(A/'source/manifest.json');assert m['source_sha256']==hashlib.sha256((E/'ge6_test1000_native_nonthinking_seed20260811/qwen3_1_7b/merged/rollouts.jsonl').read_bytes()).hexdigest()
def adjust(old,pred,target):
 a=normalized_boolean(str(target));b=normalized_boolean(pred) if pred is not None else None
 return a==b if a is not None and b is not None else bool(old)
res={};changes={}
for group in ['student64','teacher','h32','h64','h128','h256']:
 qs=[];up=down=0
 for i in range(1000):
  if group=='student64':
   rs=load(E/f'student_only64_20260910/old8/{i:04d}.json')
   for g in range(7):rs+=load(E/f'student_only64_20260910/scores/{i:04d}_{g}.json')['outputs']
   pk='predicted_answer'
  else:
   d=E/'teacher8_student8_20260909/scores'/group
   rs=load(d/f'{i:04d}.json')['outputs'] if group=='teacher' else sum([load(d/f'{i:04d}_{g}.json')['outputs'] for g in range(8)],[])
   pk='predicted'
  vals=[adjust(r['correct'],r[pk],base[i]['solution']) for r in rs]
  up+=sum(v and not r['correct'] for v,r in zip(vals,rs));down+=sum(not v and r['correct'] for v,r in zip(vals,rs))
  qs.append({'dataset_index':i,'problem_id':base[i]['problem_id'],'n':len(rs),'correct':sum(vals),'old_correct':sum(bool(r['correct']) for r in rs)})
 res[group]=qs;changes[group]={'false_to_true':up,'true_to_false':down}
 print('DONE',group,flush=True,file=__import__('sys').stderr)
anchor=lines(A/'analysis_n8x8/anchor_scores_n64.jsonl')
for x in anchor:assert base[x['dataset_index']]['problem_id']==x['problem_id']
res['anchor']=[{'dataset_index':x['dataset_index'],'problem_id':x['problem_id'],'n':64,'correct':x['correct_count_out_of_64']} for x in anchor]
print(json.dumps({'groups':res,'changes':changes,'source_manifest':m,'boolean_function':ast.unparse(fn)},ensure_ascii=False))
