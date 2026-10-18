import json,csv,os
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'pass8_results.json').read_text())['results'];data={(r['config'],r['dataset']):r for r in rs}
ds={'cruxeval_o':'CRUXEval-O','cruxeval_i':'CRUXEval-I','humaneval_plus':'HumanEval+','mbpp_plus':'MBPP+','livecodebench':'LiveCodeBench'}
configs=sorted({c for c,d in data},key=lambda c:(float(c.split('_t')[1]),int(c.split('_')[0][1:])))
baseline=[r for r in json.loads((P.parent/'full_baseline_x8_status_20260918/pass8_results.json').read_text())['results'] if r['dataset'] in ds]
data.update({(r['config'],r['dataset']):r for r in baseline})
rs=baseline+rs
configs=['student_only','teacher_only']+configs
table=[]
for c in configs:
 table.append([c]+[f"{data[c,d]['pass_at_8']*100:.2f}% / {data[c,d]['pass_all_8']*100:.2f}%" for d in ds])
lines=['# Code x8：pass@8 与 pass all 8','', '每格依次：pass@8 / pass all 8。pass@8 = 同一道题 8 次中至少 1 次通过的比例；pass all 8 = 同一道题 8 次全部通过的比例。分母为题目数，不是生成数。','', '已逐题核验 500,472 条评分（463,400 条介入配置 + 37,072 条完整题集 baseline）的题目身份、配置、replicate 0–7、评分签名；每个配置/数据集的正确次数总和与上一份准确率汇总一致。','', '| 配置 | '+' | '.join(ds.values())+' |','|---|'+'---:|'*5]
lines+=['| '+' | '.join(row)+' |' for row in table]
lines+=['','题数：CRUXEval-O/I 各 800，HumanEval+ 164，MBPP+ 378，LiveCodeBench 175。Student-only / Teacher-only 已更新为完整题集从头生成的 8 次 rollout，来源为 full_teacher_student_x8_20260918；所有行均保留生成超时计错。','', '这些是当前 8 个样本的经验统计；pass@8 表示通过测试的样本至少存在一个，并不表示实际部署能够在无判题器时选出正确样本。','', '## 各指标最佳配置','']
for d,label in ds.items():
 for metric in ['pass_at_8','pass_all_8']:
  best=max(data[c,d][metric] for c in configs);win=[c for c in configs if data[c,d][metric]==best]
  lines.append(f"- {label} {metric}: {best*100:.2f}% ({', '.join(win)})")
(P/'pass8_table.md').write_text('\n'.join(lines)+'\n')
with (P/'pass8_table.csv').open('w') as f:
 fields=['config','dataset','questions','mean_accuracy','pass_at_8_count','pass_at_8','pass_all_8_count','pass_all_8']+[f'correct_{i}_of_8_count' for i in range(9)]
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for r in rs:w.writerow({**{k:r[k] for k in fields[:8]},**{f'correct_{i}_of_8_count':n for i,n in enumerate(r['correct_count_histogram'])}})
os.environ['MPLCONFIGDIR']='/tmp/code-x8-mpl'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(17,13));fig.patch.set_facecolor('#f4f7fa');ax.axis('off')
fig.text(.04,.963,'CODE | PASS@8 & PASS ALL 8',fontsize=22,weight='bold',color='#16354a')
fig.text(.04,.935,'Each cell: at least 1 of 8 correct / all 8 correct   (percentage of questions)',fontsize=12,color='#48677a')
t=ax.table(cellText=table,colLabels=['Configuration',*ds.values()],cellLoc='center',colWidths=[.15]+[.17]*5,bbox=[0,.06,1,.9]);t.auto_set_font_size(False);t.set_fontsize(11)
for (i,j),cell in t.get_celld().items():
 cell.set_edgecolor('#dbe5ec');cell.set_linewidth(.5)
 if i==0:cell.set_facecolor('#16354a');cell.set_text_props(color='white',weight='bold')
 else:cell.set_facecolor('#e0e8ef' if i<=2 else ('#ffffff' if ((i-3)//5)%2==0 else '#eaf2f7'));cell.set_text_props(color='#173047',weight='bold' if j==0 else 'normal')
fig.text(.04,.045,'Questions per configuration: CRUXEval-O/I 800 each | HumanEval+ 164 | MBPP+ 378 | LiveCodeBench 175',fontsize=10,color='#48677a')
fig.text(.04,.025,'Student-only and teacher-only: full-dataset x8 baselines. All 500,472 code scores included; timeouts count as incorrect.',fontsize=10,color='#48677a')
plt.subplots_adjust(left=.035,right=.98,top=.93,bottom=.04)
fig.savefig(P/'pass8_table.png',dpi=160,facecolor=fig.get_facecolor())
print('\n'.join(lines))
