import json,csv,statistics,os
from pathlib import Path
P=Path(__file__).resolve().parent
x=json.loads((P/'raw_summary.json').read_text())
labels={'math/math500':'MATH500','math/olympiad_bench':'OlympiadBench','math/bbh':'BBH','code/cruxeval_o':'CRUXEval-O','code/cruxeval_i':'CRUXEval-I','code/humaneval_plus':'HumanEval+','code/mbpp_plus':'MBPP+','code/livecodebench':'LiveCodeBench'}
flat=[]
for key,z in list(x['datasets'].items())+list(x['configs'].items()):
 row={'group':key}
 for k,v in z.items():
  if isinstance(v,dict):
   for sk,sv in v.items():row[k+'_'+str(sk)]=sv
  elif isinstance(v,list):
   for i,a in enumerate(v):row[k+'_'+str(i)]=a
  else:row[k]=v
 flat.append(row)
fields=list(dict.fromkeys(k for r in flat for k in r))
with (P/'all_config_statistics.csv').open('w') as f:
 w=csv.DictWriter(f,fields);w.writeheader();w.writerows(flat)
lines=['# Teacher介入位置：完整8轮网格','', '位置p是Teacher bridge之前已经提交的response token数（0表示从输出开头介入）；相对位置p/L以该条最终输出长度为分母。首次位置仅统计至少介入一次的轨迹；未介入轨迹单独计入介入覆盖率。全部事件分布按事件加权，多介入轨迹贡献更多事件；不能解读为逐token触发概率。相对位置使用事后长度，不能直接推断因果或在线时间。','', '统计包含25配置×8轮；同一题在不同配置、轮次重复，不是独立问题数。Baseline无介入，不进入位置分布。截断/超时输出仍按保存轨迹统计。','']
for name,src in [('全部25配置（描述性汇总）',x['datasets']),('固定k=1、阈值=2', {key:x['configs'][key+'/k1_t2'] for key in labels})]:
 lines+=['## '+name,'','|数据集|轨迹数|有介入比例|次/轨迹|首次p中位数 [P25,P75]|首次p P90|首次相对位置中位数|事件前20%占比|事件后20%占比|','|---|---:|---:|---:|---|---:|---:|---:|---:|']
 for key,label in labels.items():
  a=src[key];q=a['first_position'];h=a['all_relative_decile_bins'];n=sum(h)
  lines.append(f"|{label}|{a['traces']}|{100*a['trace_intervention_rate']:.2f}%|{a['mean_events']:.2f}|{q['median']:.1f} [{q['p25']:.1f},{q['p75']:.1f}]|{q['p90']:.1f}|{100*a['first_relative']['median']:.2f}%|{100*sum(h[:2])/n:.2f}%|{100*sum(h[-2:])/n:.2f}%|")
lines+=['','## 配置敏感性（25配置范围）','','|数据集|有介入比例范围|首次p中位数范围|首次相对位置中位数范围|前20%事件占比范围|','|---|---:|---:|---:|---:|']
for key,label in labels.items():
 gs=[v for k,v in x['configs'].items() if k.startswith(key+'/')];assert len(gs)==25
 vals=[[100*g['trace_intervention_rate'] for g in gs],[g['first_position']['median'] for g in gs],[100*g['first_relative']['median'] for g in gs],[100*sum(g['all_relative_decile_bins'][:2])/g['events'] for g in gs]]
 lines.append('|'+label+'|'+'|'.join(f'{min(v):.2f}–{max(v):.2f}' for v in vals)+'|')
lines+=['','绝对位置直方图实际为[0,64),…,[512,576),[576,+∞)，尾箱合并。原始meta中的640边界笔误以此为准。','首次位置量化只表示“何时开始借助Teacher”，不能据此断言推理关键步骤。所有事件的位置反映触发与Teacher段长度共同作用；改变k会改变后续轨迹和可触发位置。']
(P/'REPORT.md').write_text('\n'.join(lines)+'\n')
os.environ['MPLCONFIGDIR']='/tmp/interv-mpl'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
fig,axes=plt.subplots(2,4,figsize=(16,7),layout='constrained')
for ax,(key,label) in zip(axes.flat,labels.items()):
 a=x['datasets'][key];b=x['configs'][key+'/k1_t2'];xx=np.arange(10)*10+5
 ax.plot(xx,100*np.array(a['all_relative_decile_bins'])/a['events'],'o-',label='All 25 configs',color='#0072B2')
 ax.plot(xx,100*np.array(b['all_relative_decile_bins'])/b['events'],'s--',label='k=1, threshold=2',color='#D55E00')
 ax.axhline(10,color='gray',linestyle=':',lw=.8);ax.set_title(label);ax.set_xlabel('Position / final output length (%)');ax.set_ylabel('Share of intervention events (%)');ax.set_xticks([10,30,50,70,90]);ax.grid(alpha=.15)
axes.flat[0].legend(fontsize=8)
fig.savefig(P/'event_position_distribution.png',dpi=200);fig.savefig(P/'event_position_distribution.svg');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
for j,field in enumerate(['first_position','first_relative']):
 for i,(key,label) in enumerate(labels.items()):
  a=x['configs'][key+'/k1_t2'][field];scale=100 if j else 1
  axes[j].plot([scale*a['p10'],scale*a['p90']],[i,i],color='#95b9ce',lw=2)
  axes[j].plot([scale*a['p25'],scale*a['p75']],[i,i],color='#0072B2',lw=7)
  axes[j].scatter(scale*a['median'],i,color='#D55E00',s=30,zorder=3)
 axes[j].set_yticks(range(8),labels.values());axes[j].invert_yaxis();axes[j].set_xlabel('First intervention: response token boundary' if j==0 else 'First intervention / final length (%)');axes[j].grid(axis='x',alpha=.2)
fig.savefig(P/'first_position_quantiles.png',dpi=200);fig.savefig(P/'first_position_quantiles.svg');plt.close(fig)
print('\n'.join(lines))
