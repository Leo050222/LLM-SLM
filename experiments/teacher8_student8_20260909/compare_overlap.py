import json
from pathlib import Path
from itertools import combinations
from statistics import mean
P=Path(__file__).resolve().parent
base=json.loads((P/'student_only_correctness.json').read_text())
q=json.loads((P/'per_question.json').read_text())
assert len(base)==8000 and len(q)==1000
by={}
for r in base:
 i=str(r['problem_id']);by.setdefault(i,{})
 assert r['replicate_index'] not in by[i]
 by[i][r['replicate_index']]=bool(r['correct'])
assert len(by)==1000 and all(set(x)==set(range(8)) for x in by.values())
assert set(by)=={str(r['problem_id']) for r in q}
S={i for i,v in by.items() if any(v.values())}
T={str(r['problem_id']) for r in q if r['teacher_correct']>0}
sets={'Student':S,'Teacher':T}
for h in (32,64,128,256):sets[f'H{h}']={str(r['problem_id']) for r in q if sum(r[f'h{h}_per_teacher_correct'])>0}
def pair(a,b):return {'intersection':len(a&b),'union':len(a|b),'jaccard_pct':100*len(a&b)/len(a|b),'only_a':len(a-b),'only_b':len(b-a)}
result={'definition':'at least one correct answer across all available samples: 8 for baselines; 64 per prefix-length arm','counts':{k:len(s) for k,s in sets.items()},'pairs':{a+' vs '+b:pair(sets[a],sets[b]) for a,b in combinations(sets,2)},'all_six_intersection':len(set.intersection(*sets.values())),'all_six_union':len(set.union(*sets.values())),'prefix_four_intersection':len(set.intersection(*(sets[f'H{h}'] for h in (32,64,128,256)))),'prefix_four_union':len(set.union(*(sets[f'H{h}'] for h in (32,64,128,256)))),'matched8':{}}
for h in (32,64,128,256):
 arms=[{str(r['problem_id']) for r in q if r[f'h{h}_per_teacher_correct'][tr]>0} for tr in range(8)]
 result['matched8'][str(h)]={'mean_covered_questions':mean(map(len,arms))}
 for name,s in [('Student',S),('Teacher',T)]:
  pp=[pair(s,a) for a in arms]
  result['matched8'][str(h)][name]={k:mean(p[k] for p in pp) for k in pp[0]}
(P/'overlap_summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
