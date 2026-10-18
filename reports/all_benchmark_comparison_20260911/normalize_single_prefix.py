"""Read-only aggregation using the existing boolean-normalization scorer."""
import ast,json,re
from pathlib import Path
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo')
E=R/'opd_eval_deepmath'
src=R/'ocrd_pipeline/full_rollout/analyze_teacher_rollout_shard.py'
fn=next(n for n in ast.parse(src.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='normalized_boolean')
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(src),'exec'))
def adjust(row):
 a=normalized_boolean(str(row['solution']))
 if a is None:return bool(row['correct'])
 pred=row['predicted_answer']; b=normalized_boolean(str(pred)) if pred is not None else None
 return a==b if a is not None and b is not None else bool(row['correct'])
base={r['dataset_index']:r for r in map(json.loads,(E/'ge6_test1000_native_nonthinking_seed20260811/qwen3_1_7b/merged/rollouts.jsonl').read_text().splitlines())}
for h in [32,64,128,256]:
 rows=json.loads((E/f'teacher_prefix_handoff_20260909/h{h}.json').read_text())
 assert len(rows)==1000 and len({r['dataset_index'] for r in rows})==1000
 for r in rows:assert r['problem_id']==base[r['dataset_index']]['problem_id'] and r['solution']==base[r['dataset_index']]['solution']
 vals=[adjust(r) for r in rows]
 print(json.dumps(dict(k=h,n=1000,correct=sum(vals),old_correct=sum(bool(r['correct']) for r in rows),false_to_true=sum(v and not r['correct'] for v,r in zip(vals,rows)),true_to_false=sum(not v and r['correct'] for v,r in zip(vals,rows)))))
