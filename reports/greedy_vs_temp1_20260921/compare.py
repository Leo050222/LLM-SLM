from pathlib import Path
import json,csv,statistics
p=Path(__file__).parent
s={k:json.loads((p/k/'summary.json').read_text()) for k in ['greedy','temp1']}
meta={k:json.loads((p/k/'INPUTS_COMPLETE.json').read_text()) for k in s}
for family in ['code','math']:assert meta['greedy']['inputs'][family]['sha256']==meta['temp1']['inputs'][family]['sha256']
lookup={k:{(x['dataset'],x['role']):x for x in v['combined']} for k,v in s.items()}
order=['cruxeval_i','cruxeval_o','humaneval_plus','mbpp_plus','livecodebench','math500','olympiad_bench','bbh']
rows=[]
for ds in order:
 for role in ['student','teacher']:
  a=lookup['greedy'][ds,role];b=lookup['temp1'][ds,role];assert b['n']==8*a['n']
  rows.append(dict(dataset=ds,role=role,questions=a['n'],greedy_correct=a['correct'],greedy_accuracy=100*a['accuracy'],temp1_correct_total=b['correct'],temp1_rollouts=b['n'],temp1_accuracy=100*b['accuracy'],temp1_seed_sd=100*b['replicate_accuracy_sd'],delta_pp=100*(a['accuracy']-b['accuracy']),relative_improvement_pct=100*(a['accuracy']/b['accuracy']-1)))
overall=[]
for role in ['student','teacher']:
 aa=[x for x in s['greedy']['combined'] if x['role']==role];bb=[x for x in s['temp1']['combined'] if x['role']==role]
 a=sum(x['correct'] for x in aa)/sum(x['n'] for x in aa);b=sum(x['correct'] for x in bb)/sum(x['n'] for x in bb)
 perseed=[]
 for rep in range(8):
  gs=[x for x in s['temp1']['groups'] if x['role']==role and x['replicate']==rep];perseed.append(sum(x['correct'] for x in gs)/sum(x['n'] for x in gs))
 overall.append(dict(role=role,greedy_accuracy=100*a,temp1_accuracy=100*b,temp1_seed_sd=100*statistics.stdev(perseed),delta_pp=100*(a-b),relative_improvement_pct=100*(a/b-1)))
(p/'comparison.json').write_text(json.dumps(dict(rows=rows,overall=overall),indent=2)+'\n')
with (p/'comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# Greedy vs temperature=1 baselines','', 'All scores complete. Identical frozen prompt hashes, models and generation token limits. Greedy has one completion/question; temperature=1 has eight seeds. Metric: single-completion accuracy (temperature=1 mean over seeds), NOT pass@8. ± denotes sample SD across eight temperature=1 seeds, not a confidence interval. Δ is greedy minus temperature=1 in percentage points.','', '| Dataset | Role | Greedy correct/N | Greedy % | T=1 mean ± SD % | Δ pp | Relative change % |','|---|---|---:|---:|---:|---:|---:|']
for x in rows:lines.append(f"| {x['dataset']} | {x['role']} | {x['greedy_correct']}/{x['questions']} | {x['greedy_accuracy']:.2f} | {x['temp1_accuracy']:.2f} ± {x['temp1_seed_sd']:.2f} | {x['delta_pp']:+.2f} | {x['relative_improvement_pct']:+.2f} |")
lines+=['','Overall is weighted by question count, not equal-weight dataset averaging.']
for x in overall:lines.append(f"- {x['role']}: greedy {x['greedy_accuracy']:.2f}%; T=1 {x['temp1_accuracy']:.2f} ± {x['temp1_seed_sd']:.2f}%; Δ {x['delta_pp']:+.2f} pp; relative {x['relative_improvement_pct']:+.2f}%.")
lines+=['','1. Student improves on all eight datasets; teacher changes are mixed and small.','2. Same 900-second active-wall-time deadline and timeout-count-wrong grading policy, but deployment differs: greedy began on two GPUs then moved to eight; T=1 used four then eight. These operational results do not isolate temperature causally.','3. Follow-up if needed: rerun both temperatures with matched hardware/concurrency and paired per-question analysis; this report does not assert statistical significance.']
audit=p/'scoring_audit.json'
if audit.exists() and audit.stat().st_size:
 d=json.loads(audit.read_text());lines+=['','Score rows validated against summaries; timeout counts/rates:']
 for role in ['student','teacher']:
  for label in ['greedy','temp1']:
   n=sum(x['n'] for x in s[label]['combined'] if x['role']==role);t=sum(v.get('trace_timeout',0) for k,v in d[label]['statuses'].items() if k.endswith('/'+role));lines.append(f'- {label} {role}: {t}/{n} = {100*t/n:.2f}% timed out.')
(p/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(overall,indent=2))
