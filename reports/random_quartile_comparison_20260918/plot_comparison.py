from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
P=Path(__file__).parent
d=json.loads((P/'r1_summary_with_teacher.json').read_text());g={x['dataset']:x for x in d['groups']}
order=['bbh','cruxeval_i','cruxeval_o','humaneval_plus','mbpp_plus','livecodebench','math500','olympiad_bench']
labels=['BBH','CruxEval-I','CruxEval-O','HumanEval+','MBPP+','LiveCodeBench','Math500','OlympiadBench']
methods=['弱模型独立生成','四段随机介入 k=1','原 R1 阈值介入 k=1, π=2','强模型直接生成']
keys=['student_accuracy','random_accuracy','threshold_r1_accuracy','teacher_accuracy']
colors=['#8291A3','#E99A46','#3874CB','#3B9C88']
plt.rcParams.update({'font.family':'Arial Unicode MS','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':'#CCD3DB','text.color':'#202D3E','axes.labelcolor':'#344256','xtick.color':'#344256','ytick.color':'#344256','pdf.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(15,9),facecolor='white');gs=fig.add_gridspec(2,1,height_ratios=[3.2,1.5],left=.07,right=.98,top=.79,bottom=.13,hspace=.38)
ax=fig.add_subplot(gs[0]);x=np.arange(8);width=.20
for j,(k,c) in enumerate(zip(keys,colors)):
 vals=[g[ds][k] for ds in order];bars=ax.bar(x+(j-1.5)*width,vals,width=.175,color=c,zorder=3)
 for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+1.1,f'{v:.1f}',ha='center',va='bottom',fontsize=9)
ax.set_xticks(x,labels);ax.set_ylim(0,100);ax.set_yticks(range(0,101,20));ax.set_ylabel('准确率（%）');ax.grid(axis='y',color='#E8ECF1',zorder=0);ax.tick_params(axis='both',length=0,pad=8)
ax.set_title('各数据集 · 单次 rollout',loc='left',fontsize=13,pad=16)
ax2=fig.add_subplot(gs[1]);pos=ax2.get_position();ax2.set_position([.25,pos.y0,.73,pos.height]);totals=[100*sum(g[ds][k.replace('_accuracy','_correct')] for ds in order)/4571 for k in keys]
for j,(v,c) in enumerate(zip(totals,colors)):
 ax2.barh(3-j,v,height=.55,color=c,zorder=3);ax2.text(v+1,3-j,f'{v:.2f}%',va='center',fontsize=12)
ax2.set_yticks([3,2,1,0],methods);ax2.set_xlim(0,100);ax2.set_xticks(range(0,101,20));ax2.set_xlabel('准确率（%）');ax2.grid(axis='x',color='#E8ECF1',zorder=0);ax2.tick_params(axis='both',length=0,pad=8)
ax2.set_title('总体 · 4,571 题按题加权',loc='left',fontsize=13,pad=13)
fig.suptitle('介入位置消融：随机介入与阈值介入',x=.07,y=.975,ha='left',fontsize=21)
fig.text(.07,.925,'Qwen3-1.7B（弱模型） / Qwen3-14B（强模型）',fontsize=12,color='#5B697B')
fig.legend(handles=[Patch(facecolor=c,label=m) for c,m in zip(colors,methods)],loc='upper left',bbox_to_anchor=(.064,.90),ncol=4,frameon=False,columnspacing=2)
fig.text(.07,.05,'统一生成超时计错；随机组计划介入 49,507 次，实际 43,468 次（保留正常 EOS）。',fontsize=10,color='#5B697B')
fig.text(.07,.025,'随机组以原 R1 为参照；弱／强模型直接生成采用历史单次基线。柱上数值为准确率，不含多种子误差区间。',fontsize=10,color='#5B697B')
for ext in ['png','pdf','svg']:fig.savefig(P/f'accuracy_comparison.{ext}',dpi=180,facecolor='white')
print(P.resolve()/'accuracy_comparison.png')
