import json,csv,collections,os,html
from pathlib import Path
P=Path(__file__).resolve().parent
s=json.loads((P/'x8_score_summary.json').read_text());gens=json.loads((P/'generation_totals.json').read_text())
ds={'cruxeval_o':'CRUXEval-O','cruxeval_i':'CRUXEval-I','humaneval_plus':'HumanEval+','mbpp_plus':'MBPP+','livecodebench':'LiveCodeBench'}
totals=collections.defaultdict(collections.Counter)
for g in gens['groups']:
 for d,v in g['datasets'].items(): totals[g['config'],d].update(v)
metrics={}
for r in s['combined']:
 c,d=r['config'],r['dataset'];a=totals[c,d];assert a['n']==r['n']
 metrics[c,d]=[100*r['accuracy'],a['interventions']/a['n'],100*a['teacher_tokens']/a['output_tokens'],a['output_tokens']/a['n']]
configs=sorted({c for c,d in metrics},key=lambda c:(float(c.split('_t')[1]),int(c.split('_')[0][1:])))
b=json.loads((P.parent.parent/'outputs/benchmark-viz/data.json').read_text())
rows=[]
for name in ['student_only','teacher_only']:
 old=next(r for r in b['rows'] if r['name']==name)
 rows.append(dict(name=name,threshold='—',k='—',scope='historical_single',values={d:old['v'][b['bench'].index(label)] for d,label in ds.items()}))
for c in configs:rows.append(dict(name=c,threshold=c.split('_t')[1],k=c.split('_')[0][1:],scope='x8',values={d:metrics[c,d] for d in ds}))
def cell(v):return f'{v[0]:.2f}% / {v[1]:.2f} / {v[2]:.2f}% / {v[3]:.2f}'
headers=['配置','阈值','k',*ds.values()]
table=[[r['name'],r['threshold'],r['k'],*[cell(r['values'][d]) for d in ds]] for r in rows]
notes='准确率按全部 8 次输出汇总；平均介入次数及平均输出长度均包含零介入与生成超时样本。Teacher token 占比 = Teacher 输出 token 总数 / 输出 token 总数，不包含 prompt 或丢弃草稿。Student/Teacher 两行是历史单次基线，其余 25 行均为完整 x8 统计。'
md=['# Code 排列组合表：8 次 rollout','', '**每格依次：准确率 / 平均介入次数 / Teacher token占比 / 平均输出token数。**','', '| '+' | '.join(headers)+' |','|'+'---|'*len(headers)]
md+=['| '+' | '.join(row)+' |' for row in table];md+=['',notes]
(P/'four_metrics_table.md').write_text('\n'.join(md)+'\n')
(P/'four_metrics_data.json').write_text(json.dumps(dict(datasets=ds,rows=rows),ensure_ascii=False,indent=2))
with (P/'four_metrics_table.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['config','threshold','k','scope','dataset','accuracy_pct','mean_interventions','teacher_token_pct','mean_output_tokens'])
 for r in rows:
  for d in ds:w.writerow([r['name'],r['threshold'],r['k'],r['scope'],ds[d],*r['values'][d]])
trs=''.join('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in row)+'</tr>' for row in table)
(P/'four_metrics_table.html').write_text('<!doctype html><meta charset="utf-8"><title>Code x8 四指标表</title><style>body{font:14px system-ui;color:#173047;background:#f4f7fa;padding:32px}h1{font-size:26px}table{border-collapse:collapse;background:white;width:100%;font-variant-numeric:tabular-nums}th{background:#16354a;color:white}th,td{padding:12px 10px;border-bottom:1px solid #dde5ec;text-align:center;white-space:nowrap}tr:nth-child(even){background:#f0f5f8}tr:nth-child(5n+3){border-top:2px solid #91afbf}p{line-height:1.7}td:first-child{font-weight:600}</style><h1>Code 排列组合表 · 8 次 rollout</h1><p>每格依次：准确率 / 平均介入次数 / Teacher token占比 / 平均输出token数</p><table><thead><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr></thead><tbody>'+trs+'</tbody></table><p>'+notes+'</p>')
os.environ['MPLCONFIGDIR']='/tmp/code-x8-mpl'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(23,15));fig.patch.set_facecolor('#f4f7fa');ax.axis('off')
fig.text(.035,.966,'CODE BENCHMARKS | 8 ROLLOUTS',fontsize=23,weight='bold',color='#16354a')
fig.text(.035,.939,'Each cell: Accuracy / Mean interventions / Teacher token share / Mean output tokens',fontsize=13,color='#48677a')
t=ax.table(cellText=table,colLabels=['Configuration','Threshold','k',*ds.values()],cellLoc='center',colWidths=[.092,.049,.028]+[.1662]*5,bbox=[0,.06,1,.90]);t.auto_set_font_size(False);t.set_fontsize(9)
for (i,j),c in t.get_celld().items():
 c.set_edgecolor('#dbe5ec');c.set_linewidth(.5)
 if i==0:c.set_facecolor('#16354a');c.set_text_props(color='white',weight='bold',fontsize=10)
 else:
  c.set_facecolor('#e7edf2' if i<=2 else ('#ffffff' if ((i-3)//5)%2==0 else '#edf4f8'))
  c.set_text_props(color='#173047',weight='bold' if j==0 else 'normal')
fig.text(.035,.05,'25 configurations x 5 datasets x 8 rollouts = 463,400 scored outputs. Student/teacher baselines: historical single rollout.',fontsize=11,color='#48677a')
fig.text(.035,.029,'Teacher share is weighted by output tokens. Means include zero-intervention and generation-timeout samples.',fontsize=11,color='#48677a')
plt.subplots_adjust(left=.03,right=.98,top=.93,bottom=.04)
fig.savefig(P/'four_metrics_table.png',dpi=170,facecolor=fig.get_facecolor())
print('Created four-metric table: markdown, CSV, JSON, HTML, PNG')
