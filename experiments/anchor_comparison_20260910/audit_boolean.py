exec(open('/dev/stdin').read()) if False else None
import json,ast,re,collections
from pathlib import Path
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');E=R/'opd_eval_deepmath';src=R/'ocrd_pipeline/full_rollout/analyze_teacher_rollout_shard.py'
t=ast.parse(src.read_text());fn=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='normalized_boolean');exec(compile(ast.Module(body=[fn],type_ignores=[]),str(src),'exec'))
def load(p):return json.loads(p.read_text())
def shape(v):
 s=v.strip().lower();m=re.fullmatch(r'\\text\{(.*)\}',s,flags=re.S)
 if m:s=m.group(1).strip()
 s=re.sub(r'^[\s$\\{}]+|[\s$\\{}.,;:!*]+$','',s)
 return 'bare' if s in ['yes','no','true','false'] else 'with_tail'
pairs=collections.Counter();cats=collections.Counter();methods=collections.Counter();examples={};questions={};target_types=collections.Counter();old_solved=new_solved=0;promoted=[]
for i in range(1000):
 rs=load(E/f'student_only64_20260910/old8/{i:04d}.json');target=rs[0]['solution'];problem=rs[0]['problem'];a=normalized_boolean(target)
 if a is not None:target_types[shape(target)]+=1
 for g in range(7):rs+=load(E/f'student_only64_20260910/scores/{i:04d}_{g}.json')['outputs']
 old=sum(bool(r['correct']) for r in rs);new=0
 for r in rs:
  pred=r['predicted_answer'];b=normalized_boolean(pred) if pred is not None else None;v=a==b if a is not None and b is not None else bool(r['correct']);new+=v
  if v and not r['correct']:
   cat=shape(target)+' / '+shape(pred);cats[cat]+=1;methods[r['score_method']]+=1;pairs[(target,pred)]+=1
   examples.setdefault(cat,[])
   if len(examples[cat])<8:examples[cat].append({'i':i,'target':target,'pred':pred,'problem':problem[:700]})
 old_solved+=old>0;new_solved+=new>0
 if new!=old:questions[i]={'old':old,'new':new,'target':target}
 if old==0 and new>0:promoted.append(i)
print(json.dumps({'changed_count':sum(cats.values()),'categories':dict(cats),'old_methods':dict(methods),'target_types_questions':dict(target_types),'affected_questions':len(questions),'old_solved':old_solved,'new_solved':new_solved,'promoted_questions':len(promoted),'top_pairs':[{'target':a,'pred':b,'count':c} for (a,b),c in pairs.most_common(30)],'examples':examples,'questions':questions},ensure_ascii=False,indent=2))
