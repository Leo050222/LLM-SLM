from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
P=Path(__file__).parent
g={r['dataset']:r for r in json.loads((P/'r1_summary_with_teacher.json').read_text())['groups']}
order=['bbh','cruxeval_i','cruxeval_o','humaneval_plus','mbpp_plus','livecodebench','math500','olympiad_bench']
labels=['BBH','CruxEval-I','CruxEval-O','HumanEval+','MBPP+','LiveCodeBench','Math500','OlympiadBench']
keys=['student_accuracy','random_accuracy','threshold_r1_accuracy','teacher_accuracy']
names=['弱模型','随机介入','阈值介入','强模型']
colors=['#8291A3','#E99A46','#3874CB','#3B9C88']
plt.rcParams.update({'font.family':'Arial Unicode MS','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':'#CCD3DB','text.color':'#202D3E','axes.labelcolor':'#344256','xtick.color':'#344256','ytick.color':'#344256','pdf.fonttype':42,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(14,4.8));fig.subplots_adjust(left=.065,right=.99,bottom=.13,top=.87)
x=np.arange(8)
for j,(key,color) in enumerate(zip(keys,colors)):
 vals=[g[ds][key] for ds in order]
 bars=ax.bar(x+(j-1.5)*.2,vals,width=.175,color=color,zorder=3)
 for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+1.25,f'{v:.1f}',ha='center',va='bottom',fontsize=9)
ax.set_xticks(x,labels);ax.set_ylim(0,100);ax.set_yticks(range(0,101,20));ax.set_ylabel('准确率 (%)');ax.grid(axis='y',color='#E8ECF1',zorder=0);ax.tick_params(axis='both',length=0,pad=8)
fig.legend(handles=[Patch(facecolor=c,label=n) for c,n in zip(colors,names)],loc='upper center',bbox_to_anchor=(.53,.99),ncol=4,frameon=False,columnspacing=2.5)
for ext in ['png','pdf','svg']:fig.savefig(P/f'accuracy_comparison_compact.{ext}',dpi=200,facecolor='white')
