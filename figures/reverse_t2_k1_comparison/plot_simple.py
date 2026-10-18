from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[2]
x=json.loads((R/'reports/reverse_t2_k1_baseline_comparison_20260918.json').read_text())
order=['cruxeval_o','cruxeval_i','humaneval_plus','mbpp_plus','livecodebench','olympiad_bench','bbh','math500']
labels=['CRUXEval-O','CRUXEval-I','HumanEval+','MBPP+','LiveCodeBench','OlympiadBench','BBH','MATH500','Overall']
rev={r['dataset']:r for r in x['reverse_summary']['groups']}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':'#CBD5E1','text.color':'#243447','xtick.color':'#526176','ytick.color':'#243447','pdf.fonttype':42,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(9,6.7));fig.subplots_adjust(left=.19,right=.97,top=.85,bottom=.11)
y=np.arange(9);h=.24
for role,color,label,offset in [('student','#78889D','Student only',-h),('teacher','#287F91','Teacher only',0),('reverse','#DE7955','Reverse (k=1, τ=2)',h)]:
 g=rev if role=='reverse' else x['baselines'][role]
 v=[100*g[d]['correct']/g[d]['n'] for d in order]+[100*sum(g[d]['correct'] for d in order)/4571]
 ax.barh(y+offset,v,h*.9,color=color,label=label,zorder=3)
 for yy,vv in zip(y+offset,v):ax.text(vv+.8,yy,f'{vv:.1f}',va='center',fontsize=9,color='#455468')
ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xticks(range(0,101,20));ax.set_xlabel('Accuracy (%)',labelpad=9);ax.tick_params(axis='y',length=0);ax.xaxis.grid(True,color='#E6EBF1',linewidth=.7,zorder=0)
ax.axhline(7.5,color='#B9C5D1',lw=.7);ax.axhspan(7.55,8.45,color='#F2F5F8',zorder=0)
fig.text(.19,.955,'Accuracy comparison',fontsize=19,fontweight='bold',ha='left')
ax.legend(loc='lower left',bbox_to_anchor=(0,1.025),ncol=3,frameon=False,fontsize=10,handlelength=1.25,columnspacing=1.5,borderaxespad=0)
for ext in ['png','pdf','svg']:fig.savefig(Path(__file__).parent/f'reverse_vs_baselines_simple.{ext}',dpi=240,facecolor='white',bbox_inches='tight')
