from pathlib import Path
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
P=Path(__file__).resolve().parent
rows=json.loads((P/'four_metrics.json').read_text());data={(r['dataset'],r['method']):r for r in rows}
font=FontProperties(fname='/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
names={'cruxeval_o':'CRUXEval-O','cruxeval_i':'CRUXEval-I','humaneval_plus':'HumanEval+','mbpp_plus':'MBPP+','livecodebench':'LiveCodeBench','olympiad_bench':'OlympiadBench','bbh':'BBH','math500':'MATH500','Overall':'总体（按题数加权）'}
methods=['student','main','teacher']
headers=['数据集','Student greedy','k = 1, π = 2 greedy','Teacher greedy']
def cell(r):return f"{r['accuracy_pct']:.2f}% / {r['mean_interventions']:.2f} / {r['teacher_token_pct']:.2f}% / {r['mean_output_tokens']:.2f}"
values=[[label]+[cell(data[ds,m]) for m in methods] for ds,label in names.items()]
fig,ax=plt.subplots(figsize=(16.8,6.6));fig.patch.set_facecolor('white');ax.axis('off')
fig.text(.035,.945,'Greedy 解码：准确率与生成成本',fontproperties=font,fontsize=21,color='#14263d')
fig.text(.035,.886,'每格：acc / 平均介入次数 / Teacher token占比 / 平均输出token数',fontproperties=font,fontsize=13,color='#506070')
t=ax.table(cellText=values,colLabels=headers,cellLoc='center',colWidths=[.19,.27,.27,.27],bbox=[0,.13,1,.72])
t.auto_set_font_size(False)
for (i,j),c in t.get_celld().items():
 c.set_text_props(fontproperties=font,fontsize=11.5,color='#162639');c.set_edgecolor('#dce3e9');c.set_linewidth(.6)
 if i==0:c.set_facecolor('#19364f');c.set_text_props(color='white',fontsize=13)
 else:
  c.set_facecolor('#eaf3fb' if j==2 else ('#f5f7fa' if i%2==0 else 'white'))
  if i==9:c.set_facecolor('#dceaf2');c.set_linewidth(1.0)
fig.text(.035,.093,'8 个数据集，4,571 题，每题 1 次。包含超时保留输出；acc 将超时计错。',fontproperties=font,fontsize=10.5,color='#506070')
fig.text(.035,.052,'Teacher token占比 = 教师输出token总数 / 最终输出token总数；不含prompt与丢弃草稿。Teacher-only 无切换介入，介入次数记为0。',fontproperties=font,fontsize=10.5,color='#506070')
fig.subplots_adjust(left=.03,right=.97,top=.94,bottom=.02)
for ext in ['png','pdf']:fig.savefig(P/f'four_metrics_table.{ext}',dpi=220,facecolor='white')
with (P/'four_metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['| '+' | '.join(headers)+' |','|---|---|---|---|']+['| '+' | '.join(v)+' |' for v in values]
(P/'four_metrics_table.md').write_text('每格：acc / 平均介入次数 / Teacher token占比 / 平均输出token数\n\n'+'\n'.join(lines)+'\n')
# Check accuracy and intervention totals against existing verified summary.
for r in json.loads((P/'summary.json').read_text())['groups']:
 z=data[r['dataset'],'main'];assert z['correct']==r['correct'] and z['n']==r['n'] and z['interventions']==r['actual_takeovers_total']
print('Saved PNG, PDF, CSV and Markdown; all main summary counts match.')
