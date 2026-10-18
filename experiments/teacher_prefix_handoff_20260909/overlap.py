import json
from pathlib import Path
from itertools import combinations
root=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath')
out=root/'teacher_prefix_handoff_20260909'
rows={}
for name,model in [('Student','qwen3_1_7b'),('Teacher','qwen3_14b')]:
 rows[name]=[json.loads(x) for x in (root/f'ge6_test1000_native_nonthinking_seed20260811/{model}/merged/rollouts.jsonl').read_text().splitlines() if x.strip()]
for h in (32,64,128,256):rows[f'H{h}']=json.loads((out/f'h{h}.json').read_text())
ids=None
sets={}
for name,rs in rows.items():
 mapping={str(r['problem_id']):r for r in rs}
 assert len(rs)==len(mapping)==1000
 if ids is None:ids=set(mapping)
 assert ids==set(mapping)
 sets[name]={i for i,r in mapping.items() if r['correct']}
pairs=[]
for a,b in combinations(sets,2):
 sa,sb=sets[a],sets[b]
 inter=sa&sb
 pairs.append(dict(a=a,b=b,intersection=len(inter),union=len(sa|sb),jaccard=100*len(inter)/len(sa|sb),a_correct_retained=100*len(inter)/len(sa),b_correct_shared=100*len(inter)/len(sb),only_a=len(sa-sb),only_b=len(sb-sa)))
hs=[sets[f'H{h}'] for h in (32,64,128,256)]
result={'counts':{k:len(v) for k,v in sets.items()},'pairs':pairs,'four_prefixes':{'intersection':len(set.intersection(*hs)),'union':len(set.union(*hs)),'correct_in_exactly_k':{k:sum(sum(i in s for s in hs)==k for i in ids) for k in range(5)}},'all_six':{'intersection':len(set.intersection(*sets.values())),'union':len(set.union(*sets.values()))},'prefix_union_outside_student':len(set.union(*hs)-sets['Student']),'prefix_union_outside_teacher':len(set.union(*hs)-sets['Teacher'])}
print(json.dumps(result,indent=2))
