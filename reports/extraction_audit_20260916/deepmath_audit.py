import json,re,collections
from pathlib import Path
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath')
out=[]
for group in ['online_chunk64_k8_test1000_2nodes_seed20260911','online_chunk64_k32_test1000_h100_seed20260911']:
 c=collections.Counter()
 for f in (R/group/'results/completed').glob('*.json'):
  x=json.loads(f.read_text());c['n']+=1;s=x['response'].rsplit('</think>',1)[-1].strip();boxes=[]
  for m in re.finditer(r'\\boxed\s*\{',s):
   depth=1;end=m.end()
   while end<len(s) and depth:
    depth+=(s[end]=='{')-(s[end]=='}');end+=1
   if depth==0:boxes.append(s[m.end():end-1])
  pred=boxes[-1].strip() if boxes else None
  if boxes:c['boxed']+=1
  else:
   tail=s[-1200:];markers=list(re.finditer(r'(?is)(?:final\s+answer|answer\s+is|answer\s*:|therefore[, ]+the\s+answer\s+is|thus[, ]+the\s+answer\s+is)\s*[:=]?\s*',tail))
   if markers:
    segment=tail[markers[-1].end():].strip();spans=re.findall(r'\$\$([^$]+)\$\$|\$([^$]+)\$',segment,flags=re.S)
    if spans:pred=next(v for v in reversed(spans[-1]) if v)
    else:
     lines=[v.strip() for v in segment.splitlines() if v.strip()];pred=lines[-1] if lines else None
    c['marker']+=1
  if pred is not None and pred.strip().rstrip('.* '):c['nonempty']+=1
  else:c['missing_or_empty']+=1
 out.append(dict(group=group,**c))
print(json.dumps(out))
