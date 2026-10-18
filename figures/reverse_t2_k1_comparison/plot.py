from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[2]
data=json.loads((R/'reports/reverse_t2_k1_baseline_comparison_20260918.json').read_text())
order=['cruxeval_o','cruxeval_i','humaneval_plus','mbpp_plus','livecodebench','olympiad_bench','bbh','math500']
labels=['CRUXEval-O','CRUXEval-I','HumanEval+','MBPP+','LiveCodeBench','OlympiadBench','BBH','MATH500','Overall*']
rev={x['dataset']:x for x in data['reverse_summary']['groups']}
values=[]
for role in ['student','teacher','reverse']:
 g=rev if role=='reverse' else data['baselines'][role]
 v=[100*g[d]['correct']/g[d]['n'] for d in order]
 v.append(100*sum(g[d]['correct'] for d in order)/4571);values.append(np.array(v))
s,t,r=values
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':'#CBD5E1','axes.labelcolor':'#334155','text.color':'#172A3A','xtick.color':'#475569','ytick.color':'#172A3A','pdf.fonttype':42,'svg.fonttype':'none'})
fig,(ax,bx)=plt.subplots(1,2,figsize=(15,8.5),gridspec_kw={'width_ratios':[1.35,1]},layout=None)
fig.subplots_adjust(left=.12,right=.97,top=.79,bottom=.16,wspace=.28)
colors=['#78889D','#287F91','#DE7955'];names=['Student only · 1.7B','Teacher only · 14B','Reverse · 14B + 1.7B intervention']
y=np.arange(9);h=.23
for v,c,name,off in zip(values,colors,names,[-h,0,h]):
 bars=ax.barh(y+off,v,h*.92,color=c,label=name,zorder=3)
 for bar,n in zip(bars,v):ax.text(n+.85,bar.get_y()+bar.get_height()/2,f'{n:.1f}',va='center',fontsize=8.5,color='#334155')
ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Accuracy (%)');ax.set_title('A   Accuracy across all eight datasets',loc='left',pad=17,fontweight='bold');ax.xaxis.grid(True,color='#E5EAF0',zorder=0);ax.tick_params(axis='y',length=0)
for delta,c,off,name in [(r-s,colors[0],-.15,'Reverse − student only'),(r-t,colors[1],.15,'Reverse − teacher only')]:
 bars=bx.barh(y+off,delta,.27,color=c,zorder=3,label=name)
 for bar,n in zip(bars,delta):bx.text(n+(.55 if n>=0 else -.55),bar.get_y()+bar.get_height()/2,f'{n:+.1f}',va='center',ha='left' if n<0 else 'left',fontsize=9,color='#334155') if n>=0 else bx.text(n-.55,bar.get_y()+bar.get_height()/2,f'{n:+.1f}',va='center',ha='right',fontsize=9,color='#334155')
bx.set_yticks(y,labels);bx.invert_yaxis();bx.set_xlim(-45,12);bx.set_xticks([-40,-30,-20,-10,0,10]);bx.axvline(0,color='#8895A7',lw=1);bx.xaxis.grid(True,color='#E5EAF0',zorder=0);bx.tick_params(axis='y',length=0);bx.set_xlabel('Accuracy difference (percentage points)');bx.set_title('B   Effect of reverse intervention',loc='left',pad=17,fontweight='bold')
for a in (ax,bx):
 a.axhline(7.5,color='#AAB6C5',linewidth=.8)
 a.axhspan(7.55,8.45,color='#F2F5F8',zorder=0)
ax.legend(loc='lower left',bbox_to_anchor=(0,1.08),frameon=False,fontsize=9,ncol=1)
bx.legend(loc='lower left',bbox_to_anchor=(0,1.08),frameon=False,fontsize=9,ncol=1)
fig.suptitle('Strong-model reasoning with weak-model intervention',x=.12,y=.98,ha='left',fontsize=21,fontweight='bold')
fig.text(.12,.935,'Threshold > 2  ·  k = 1  ·  One rollout per question  ·  4,571 questions',fontsize=12,color='#53667B')
fig.text(.12,.09,'* Overall is weighted by question count. Generation timeouts count as incorrect (reverse: 244 / 4,571).',fontsize=10,color='#53667B')
fig.text(.12,.062,'Historical single-rollout baselines; not an x8 mean. Reverse seed: 20260912; code baseline seed: 20260911.',fontsize=10,color='#53667B')
fig.text(.12,.034,'Matched questions/prompts; same math judge prompt and code test harness. No uncertainty intervals estimated.',fontsize=10,color='#53667B')
out=Path(__file__).parent
for ext in ['png','pdf','svg']:fig.savefig(out/f'reverse_vs_baselines.{ext}',dpi=220,facecolor='white',bbox_inches='tight')
print(out/'reverse_vs_baselines.png')
