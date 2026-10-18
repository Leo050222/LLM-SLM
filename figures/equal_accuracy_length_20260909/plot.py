from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
OUT=Path(__file__).resolve().parent
font=FontProperties(fname='/System/Library/Fonts/Hiragino Sans GB.ttc').get_name()
plt.rcParams.update({'font.family':[font,'DejaVu Sans'],'font.size':11,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'text.color':'#243247','axes.labelcolor':'#52627A','xtick.color':'#52627A','ytick.color':'#52627A','axes.edgecolor':'#D9E0E8'})
BLUE,ORANGE,GRAY='#4578B9','#D58C3E','#BFC9D6'
def heading(fig,title,sub):
 fig.text(.06,.95,title,fontsize=22,weight='bold',va='top')
 fig.text(.06,.885,sub,fontsize=11,color='#65748A',va='top')
def save(fig,name):
 fig.savefig(OUT/name,dpi=300,facecolor='white');plt.close(fig)
def grid(ax):
 ax.grid(axis='y',color='#E9EDF3');ax.set_axisbelow(True);ax.tick_params(length=0)

fig=plt.figure(figsize=(14,8))
heading(fig,'正确率相同，Teacher 平均仍多生成 140.16 tokens','3,917 个 clean Prefix · 每个 Prefix 的 Teacher 与 Student 正确数相同')
ax=fig.add_axes([.08,.29,.4,.48])
x=np.arange(2)
a=ax.bar(x-.18,[966.68,850.38],width=.34,color=BLUE,label='Student')
b=ax.bar(x+.18,[1106.84,959.13],width=.34,color=ORANGE,label='Teacher')
for bar in list(a)+list(b):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+22,f'{bar.get_height():,.2f}',ha='center',fontsize=12)
ax.set_xticks(x,['平均长度','长度中位数']);ax.set_ylabel('Continuation 长度（tokens）');ax.set_ylim(0,1350);grid(ax)
ax.legend(frameon=False,ncol=2,loc='upper center')
pie=fig.add_axes([.6,.29,.31,.49])
pie.pie([1342,2575],colors=[BLUE,ORANGE],startangle=90,autopct='%1.2f%%',pctdistance=.66,textprops={'fontsize':18,'color':'white'})
pie.set_title('Teacher 相对 Student 的长度方向',fontsize=13,pad=8)
pie.legend(handles=[Patch(color=BLUE,label='更短（1,342）'),Patch(color=ORANGE,label='更长（2,575）')],loc='lower center',bbox_to_anchor=(.5,-.09),frameon=False,ncol=2,fontsize=10)
fig.text(.08,.18,'配对长度差中位数  +91.38 tokens',fontsize=14,weight='bold')
fig.text(.08,.125,'配对差 P25 / P75：−62.50 / +322.25 tokens',fontsize=11,color='#65748A')
fig.text(.6,.16,'Pearson r = 0.806  |  Spearman ρ = 0.826',fontsize=11)
fig.text(.6,.115,'配对 t 检验与 Wilcoxon 检验：p < 10⁻⁹⁰',fontsize=11)
fig.text(.06,.045,'注：均值、中位数、相关系数与检验结果均来自提供的汇总数据；长度完全相同的 Prefix 为 0。',fontsize=10,color='#65748A')
save(fig,'01_equal_accuracy_overview.png')

counts=np.array([42,109,249,404,538,0,667,709,645,427,127])
assert counts.sum()==3917 and counts[:5].sum()==1342 and counts[6:].sum()==2575
labels=['≤ −1,000','−1,000～−500','−500～−250','−250～−100','−100～0','= 0','0～+100','+100～+250','+250～+500','+500～+1,000','≥ +1,000']
fig=plt.figure(figsize=(14,7.5))
heading(fig,'相同正确率组中，65.74% 的 Prefix 在 Teacher 下更长','长度差 ΔL = Teacher − Student · 负值表示 Teacher 更短，正值表示 Teacher 更长')
ax=fig.add_axes([.08,.25,.87,.55])
pcts=counts/counts.sum()*100
bars=ax.bar(range(11),pcts,color=[BLUE]*5+[GRAY]+[ORANGE]*5,width=.72)
for b,n,p in zip(bars,counts,pcts):ax.text(b.get_x()+b.get_width()/2,p+.4,f'{p:.2f}%\n(n={n:,})',ha='center',fontsize=10)
ax.set_xticks(range(11),labels,rotation=28,ha='right',fontsize=10)
ax.set_ylim(0,22);ax.set_ylabel('Prefix 占比（%）');grid(ax)
fig.legend(handles=[Patch(color=BLUE,label='Teacher 更短'),Patch(color=ORANGE,label='Teacher 更长')],loc='lower center',bbox_to_anchor=(.5,.1),ncol=2,frameon=False)
fig.text(.06,.045,'注：柱形对应离散区间，宽度不代表 token 跨度；百分比按 3,917 个 Prefix 重新计算。',fontsize=10,color='#65748A')
save(fig,'02_equal_accuracy_length_distribution.png')

n=np.array([1811,198,116,81,96,102,107,239,1167])
assert n.sum()==3917
student=np.array([1116.23,1208.32,1171.99,995.85,1109.12,1086.60,992.96,895.18,661.22])
teacher=np.array([1298.43,1369.61,1375.80,1190.03,1271.66,1204.20,1137.11,1026.40,724.07])
diff=np.array([182.20,161.30,203.80,194.18,162.54,117.59,144.15,131.23,62.85])
longer=np.array([67.81,66.16,71.55,71.60,66.67,73.53,63.55,63.18,61.44])
fig=plt.figure(figsize=(15,10))
heading(fig,'各共同正确数组中，Teacher 的平均长度均更长','上：两模型平均 continuation 长度；下：平均长度差与 Teacher 更长的 Prefix 占比')
x=np.arange(9)
ax=fig.add_axes([.08,.57,.87,.24])
ax.plot(x,student,color=BLUE,marker='o',lw=2,label='Student')
ax.plot(x,teacher,color=ORANGE,marker='o',lw=2,label='Teacher')
for i,(s,t) in enumerate(zip(student,teacher)):
 ax.plot([i,i],[s,t],color='#D8DFE8',lw=1,zorder=0)
 ax.text(i,s-35,f'{s:,.0f}',ha='center',va='top',fontsize=10,color=BLUE)
 ax.text(i,t+30,f'{t:,.0f}',ha='center',fontsize=10,color=ORANGE)
ax.set_xlim(-.6,8.6);ax.set_ylim(520,1520);ax.set_ylabel('平均长度（tokens）');ax.set_xticks(x,[]);grid(ax)
ax.legend(loc='upper right',frameon=False,ncol=2)
ax2=fig.add_axes([.08,.23,.87,.24])
bars=ax2.bar(x,diff,color=ORANGE,width=.55)
for i,d in enumerate(diff):ax2.text(i,d+5,f'+{d:.2f}',ha='center',fontsize=11,weight='bold')
ax2.set_xlim(-.6,8.6);ax2.set_ylim(0,255);ax2.set_ylabel('Teacher − Student（tokens）');grid(ax2)
ax2.set_xticks(x,[f'{i}/8\nn={nn:,}' for i,nn in enumerate(n)])
ax2.set_xlabel('两模型共同正确数 / 8',labelpad=9)
for i,p in enumerate(longer):ax2.text(i,12,f'{p:.2f}%',ha='center',fontsize=10,color='white')
fig.text(.08,.125,'柱内白字：Teacher 更长的 Prefix 占比。两端组：共同 0/8 时 +182.20 tokens；共同 8/8 时 +62.85 tokens。',fontsize=11)
fig.text(.08,.075,'注：点标签取整；平均差沿用提供的统计值，可能与已四舍五入的两模型均值之差相差 0.01 token。',fontsize=10,color='#65748A')
fig.text(.08,.045,'这些分组结果描述长度与正确率的关联；无法仅凭汇总数据判断额外长度的原因或因果作用。',fontsize=10,color='#65748A')
save(fig,'03_length_by_shared_accuracy.png')
print('Created 3 PNG files:',OUT)
