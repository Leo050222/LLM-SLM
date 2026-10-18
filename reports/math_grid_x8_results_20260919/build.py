import json,csv,collections,statistics,os,html
from pathlib import Path
P=Path(__file__).resolve().parent
raw=json.loads((P/'verified_aggregates.json').read_text());groups=raw['groups']
ds={'math500':('MATH500',500),'olympiad_bench':('OlympiadBench',674),'bbh':('BBH',1080)}
configs=['student_only','teacher_only']+sorted({g['config'] for g in groups if g['kind']!='baseline'},key=lambda c:(float(c.split('_t')[1]),int(c.split('_')[0][1:])))
results=[];replicates=[];question_rows=[]
for c in configs:
 gs=[g for g in groups if g['config']==c];assert len(gs)==8 and {g['replicate'] for g in gs}==set(range(8))
 for d,(label,nq) in ds.items():
  totals=collections.Counter();per=collections.defaultdict(dict);acc=[]
  for g in gs:
   v=g['groups'][d];assert v['n']==nq;totals.update(v);acc.append(v['correct']/v['n'])
   replicates.append(dict(config=c,dataset=d,replicate=g['replicate'],**v,accuracy_pct=100*v['correct']/v['n'],mean_interventions=v['interventions']/v['n'],teacher_token_pct=100*v['teacher_tokens']/v['output_tokens'],mean_output_tokens=v['output_tokens']/v['n']))
   for dataset,pid,correct in g['questions']:
    if dataset==d:assert g['replicate'] not in per[pid];per[pid][g['replicate']]=correct
  assert len(per)==nq
  h=[0]*9
  for pid,v in per.items():
   assert set(v)==set(range(8));h[sum(v.values())]+=1
   question_rows.append(dict(config=c,dataset=d,problem_id=pid,correct_count=sum(v.values()),**{f'r{i}':v[i] for i in range(8)}))
  assert sum(i*v for i,v in enumerate(h))==totals['correct']
  z=dict(config=c,dataset=d,questions=nq,rollouts=totals['n'],correct=totals['correct'],accuracy_pct=100*totals['correct']/totals['n'],replicate_accuracy_sd_pp=100*statistics.stdev(acc),replicate_accuracy_min_pct=100*min(acc),replicate_accuracy_max_pct=100*max(acc),mean_interventions=totals['interventions']/totals['n'],teacher_token_pct=100*totals['teacher_tokens']/totals['output_tokens'],mean_output_tokens=totals['output_tokens']/totals['n'],teacher_tokens=totals['teacher_tokens'],output_tokens=totals['output_tokens'],interventions=totals['interventions'],pass_at_8_count=nq-h[0],pass_at_8_pct=100*(nq-h[0])/nq,pass_all_8_count=h[8],pass_all_8_pct=100*h[8]/nq,never_pass_count=h[0],generation_timeouts=totals['generation_timeouts'],generation_timeout_pct=100*totals['generation_timeouts']/totals['n'],length_stops=totals['length_stops'],answer_correct=totals['answer_correct'],timeout_answer_correct=totals['timeout_answer_correct'],correct_count_histogram=h)
  results.append(z)
data={(r['config'],r['dataset']):r for r in results}
for r in results:
 for role in ['student','teacher']:
  b=data[role+'_only',r['dataset']]['accuracy_pct'];r[f'delta_vs_{role}_pp']=r['accuracy_pct']-b;r[f'relative_gain_vs_{role}_pct']=100*(r['accuracy_pct']-b)/b
assert len(results)==81 and sum(r['rollouts'] for r in results)==486864
(P/'summary.json').write_text(json.dumps(dict(verified_rows=486864,results=results),ensure_ascii=False,indent=2))
def csvwrite(path,rows):
 with (P/path).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
csvwrite('all_metrics.csv',[{**{k:v for k,v in r.items() if k!='correct_count_histogram'},**{f'correct_{i}_of_8_count':v for i,v in enumerate(r['correct_count_histogram'])}} for r in results]);csvwrite('replicate_metrics.csv',replicates);csvwrite('per_question_correctness.csv',question_rows)
headers=['配置',*[v[0] for v in ds.values()]]
four=[];passes=[];variability=[]
for c in configs:
 four.append([c]+[f"{data[c,d]['accuracy_pct']:.2f}% / {data[c,d]['mean_interventions']:.2f} / {data[c,d]['teacher_token_pct']:.2f}% / {data[c,d]['mean_output_tokens']:.2f}" for d in ds])
 passes.append([c]+[f"{data[c,d]['pass_at_8_pct']:.2f}% / {data[c,d]['pass_all_8_pct']:.2f}%" for d in ds])
 variability.append([c]+[f"{data[c,d]['accuracy_pct']:.2f} ± {data[c,d]['replicate_accuracy_sd_pp']:.2f} / {data[c,d]['generation_timeouts']}" for d in ds])
def mdtable(table):return '\n'.join(['| '+' | '.join(headers)+' |','|---|'+'---:|'*3]+['| '+' | '.join(row)+' |' for row in table])
notes=['所有27行均为完整题集8次rollout；Student Qwen3-1.7B、Teacher Qwen3-14B。数学网格25配置共450,800条，baseline共36,064条，总计486,864条。','准确率按全部输出汇总；标准差为8轮准确率的样本标准差，不是置信区间。pass@8按题统计至少1次正确，pass all 8按题统计8次全部正确。','Teacher token占比为Teacher输出token总数/输出token总数；平均介入次数和输出长度包含零介入与超时样本，不含prompt或丢弃草稿。','评分使用已有GPT-4o-mini判断，统一将生成超时计错。首轮使用历史提取/重评分流程，新增7轮使用后续评分流程；两者不是同一评分签名。首轮部分配置未设900秒截止，后续轮次设900秒截止，不同部署并发也可能影响超时发生率。','新增七轮中有1条已记录的judge修复：frequency_penalty=0.5；保留该条独立评分签名，未新增评分请求。数学judge结果不等同于人工真值。']
for filename,title,table in [('four_metrics_table.md','每格：准确率 / 平均介入次数 / Teacher token占比 / 平均输出token数',four),('pass8_table.md','每格：pass@8 / pass all 8',passes),('accuracy_sd_timeouts.md','每格：准确率均值 ± 标准差（百分点） / 生成超时数',variability)]:
 (P/filename).write_text('# Math/reasoning ×8\n\n'+title+'\n\n'+mdtable(table)+'\n\n'+'\n\n'.join(notes)+'\n')
report=['# Math/reasoning 全量 ×8 汇总','',*notes,'','## 最优介入配置','', '| 数据集 | 指标 | 最优值 | 配置 |','|---|---|---:|---|']
for d,(label,nq) in ds.items():
 for metric in ['accuracy_pct','pass_at_8_pct','pass_all_8_pct']:
  best=max(data[c,d][metric] for c in configs[2:]);win=[c for c in configs[2:] if data[c,d][metric]==best]
  report.append(f"| {label} | {metric} | {best:.2f}% | {', '.join(win)} |")
report+=['','## 四指标完整表','',mdtable(four),'','## Pass@8 / pass all 8','',mdtable(passes),'','## 准确率波动与超时','',mdtable(variability),'','数据文件：all_metrics.csv（81行完整指标与baseline差值），replicate_metrics.csv（648行逐轮指标），per_question_correctness.csv（60,858行逐题8轮正确性）。跨数据集不合并成一个总准确率；网格最优值为观察到的最大值，不代表显著优于其他配置。']
(P/'REPORT.md').write_text('\n'.join(report)+'\n')
sections=[]
for title,tab in [('准确率 / 平均介入次数 / Teacher token占比 / 平均输出token数',four),('pass@8 / pass all 8',passes),('准确率均值 ± 标准差 / 生成超时数',variability)]:
 sections.append('<h2>'+title+'</h2><table><tr>'+''.join('<th>'+x+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in tab)+'</table>')
(P/'tables.html').write_text('<!doctype html><meta charset="utf-8"><title>Math ×8</title><style>body{font:14px system-ui;margin:32px;color:#173047;background:#f4f7fa}table{border-collapse:collapse;width:100%;background:white;font-variant-numeric:tabular-nums}td,th{padding:11px;text-align:center;border-bottom:1px solid #dbe5ec;white-space:nowrap}th{background:#16354a;color:white}tr:nth-child(even){background:#edf4f8}h2{margin-top:40px}p{line-height:1.8}</style><h1>Math / Reasoning · 8 Rollouts</h1>'+''.join(sections)+''.join('<p>'+html.escape(x)+'</p>' for x in notes))
os.environ['MPLCONFIGDIR']='/tmp/math-x8-mpl'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
for filename,title,sub,table in [('four_metrics_table.png','MATH / REASONING | 8 ROLLOUTS','Accuracy / Mean interventions / Teacher token share / Mean output tokens',four),('pass8_table.png','MATH / REASONING | PASS@8 & PASS ALL 8','At least 1 of 8 correct / All 8 correct (percentage of questions)',passes),('accuracy_sd_timeouts.png','MATH / REASONING | VARIABILITY & TIMEOUTS','Mean accuracy +/- replicate SD (pp) / Generation timeout count',variability)]:
 fig,ax=plt.subplots(figsize=(16,14));fig.patch.set_facecolor('#f4f7fa');ax.axis('off')
 fig.text(.04,.965,title,fontsize=21,weight='bold',color='#16354a');fig.text(.04,.94,sub,fontsize=12,color='#48677a')
 t=ax.table(cellText=table,colLabels=['Configuration',*[v[0] for v in ds.values()]],cellLoc='center',colWidths=[.16,.28,.28,.28],bbox=[0,.065,1,.89]);t.auto_set_font_size(False);t.set_fontsize(10.5)
 for (i,j),cell in t.get_celld().items():
  cell.set_edgecolor('#dbe5ec');cell.set_linewidth(.5)
  if i==0:cell.set_facecolor('#16354a');cell.set_text_props(color='white',weight='bold')
  else:cell.set_facecolor('#e0e8ef' if i<=2 else ('white' if ((i-3)//5)%2==0 else '#edf4f8'));cell.set_text_props(color='#173047',weight='bold' if j==0 else 'normal')
 fig.text(.04,.048,'Full datasets: MATH500 500 | OlympiadBench 674 | BBH 1,080. All rows include 8 rollouts.',fontsize=10,color='#48677a')
 fig.text(.04,.029,'486,864 outputs verified. GPT-4o-mini grades; generation timeouts count as incorrect. Historical r0 policy differs.',fontsize=10,color='#48677a')
 plt.subplots_adjust(left=.035,right=.98,top=.93,bottom=.04);fig.savefig(P/filename,dpi=170,facecolor=fig.get_facecolor());plt.close(fig)
print('\n'.join(report[:21]))
print('TOTAL_TIMEOUTS',sum(r['generation_timeouts'] for r in results),'GRID_TIMEOUTS',sum(r['generation_timeouts'] for r in results if r['config'] not in configs[:2]))
for d in ds:
 print('BASELINES',d,[(c,{k:data[c,d][k] for k in ['accuracy_pct','pass_at_8_pct','pass_all_8_pct']}) for c in configs[:2]])
