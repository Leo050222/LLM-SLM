import json,collections
from pathlib import Path
R=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent
load=lambda p:json.loads((R/p).read_text())
m={(r['config'],r['dataset']):r for r in load('reports/math_grid_x8_results_20260919/summary.json')['results']}
c={r['name']:r for r in load('reports/code_grid_x8_results_20260918/four_metrics_data.json')['rows']}
p={(r['config'],r['dataset']):r for r in load('reports/code_grid_x8_results_20260918/pass8_results.json')['results']+load('reports/full_baseline_x8_status_20260918/pass8_results.json')['results']}
b=collections.defaultdict(collections.Counter)
for g in load('reports/full_baseline_x8_status_20260918/code_tokens.json'):
 for d,v in g['totals'].items():b[g['role']+'_only',d].update(v)
ds=['math500','olympiad_bench','bbh','cruxeval_o','cruxeval_i','humaneval_plus','mbpp_plus','livecodebench']
configs=['student_only','teacher_only']+sorted([x for x in c if '_only' not in x],key=lambda c:(float(c.split('_t')[1]),int(c.split('_')[0][1:])))
lines=['| 实验 | 阈值 | k | MATH500 | OlympiadBench | BBH | CRUXEval-O | CRUXEval-I | HumanEval+ | MBPP+ | LiveCodeBench |','| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |']
for cfg in configs:
 row=[cfg,'—' if '_only' in cfg else cfg.split('_t')[1],'—' if '_only' in cfg else cfg.split('_')[0][1:]]
 for d in ds:
  if (cfg,d) in m:
   r=m[cfg,d];v=[r[k] for k in ['accuracy_pct','pass_at_8_pct','pass_all_8_pct','mean_interventions','teacher_token_pct','mean_output_tokens']]
  else:
   r=p[cfg,d]
   if '_only' in cfg:
    a=b[cfg,d];assert a['n']==r['questions']*8
    v=[100*r['mean_accuracy'],100*r['pass_at_8'],100*r['pass_all_8'],0,100*a['teacher_tokens']/a['output_tokens'],a['output_tokens']/a['n']]
   else:
    a=c[cfg]['values'][d];v=[a[0],100*r['pass_at_8'],100*r['pass_all_8'],*a[1:]]
  row.append(' / '.join(f'{x:.2f}'+('%' if i in [0,1,2,4] else '') for i,x in enumerate(v)))
 lines.append('| '+' | '.join(row)+' |')
(P/'all_benchmarks_six_metrics.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
