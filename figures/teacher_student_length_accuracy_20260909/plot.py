from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parent
FONT = FontProperties(fname='/System/Library/Fonts/Hiragino Sans GB.ttc').get_name()
plt.rcParams.update({'font.family': [FONT, 'DejaVu Sans'], 'font.size': 11, 'axes.unicode_minus': False,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
    'axes.edgecolor': '#D6DCE5', 'text.color': '#202C40', 'axes.labelcolor': '#45546B',
    'xtick.color': '#45546B', 'ytick.color': '#45546B', 'savefig.dpi': 300})
BLUE, ORANGE, GREEN, GRAY, RED = '#4578B9', '#D58C3E', '#258C79', '#BFC9D6', '#D66970'
counts = np.array([136,571,1084,1520,1682,2,1954,2283,2201,1428,373])
labels = ['< −1,000','−1,000～−500','−500～−250','−250～−100','−100～0','0','0～+100','+100～+250','+250～+500','+500～+1,000','≥ +1,000']
total = counts.sum()
short = np.array([[85,42,9],[421,109,41],[740,249,95],[934,404,182],[947,538,197]])
long = np.array([[1025,667,262],[1204,709,370],[1165,645,391],[736,427,265],[188,127,58]])
assert total == 13234
assert np.array_equal(short.sum(1), counts[:5])
assert np.array_equal(long.sum(1), counts[6:])
assert np.array_equal(short.sum(0), [3127,1342,524])
assert np.array_equal(long.sum(0), [4318,2575,1346])

def heading(fig,title,subtitle):
    fig.text(.06,.953,title,fontsize=22,weight='bold',va='top')
    fig.text(.06,.9,subtitle,fontsize=11,color='#65748A',va='top')

def save(fig,name):
    fig.savefig(OUT/name,facecolor='white')
    plt.close(fig)

fig = plt.figure(figsize=(14,8))
heading(fig,'Teacher 更长的 Prefix 占 62.26%',f'长度差 ΔL = Teacher continuation − Student continuation · 共 {total:,} 个 Prefix')
ax = fig.add_axes([.08,.24,.87,.57])
pct = counts/total*100
bars=ax.bar(range(11),pct,color=[BLUE]*5+[GRAY]+[ORANGE]*5,width=.72,zorder=3)
for b,n,p in zip(bars,counts,pct):
    ax.text(b.get_x()+b.get_width()/2,p+.35,f'{p:.2f}%\n(n={n:,})',ha='center',va='bottom',fontsize=9)
ax.set_xticks(range(11),labels,rotation=25,ha='right',fontsize=10)
ax.set_ylim(0,21); ax.set_ylabel('Prefix 占比（%）'); ax.grid(axis='y',color='#E9EDF3'); ax.set_axisbelow(True)
ax.axvline(5,color='#CDD5DF',ls=':',lw=1)
fig.legend(handles=[Patch(color=BLUE,label=f'Teacher 更短：4,993（{4993/total*100:.2f}%）'),Patch(color=GRAY,label='相同：2（0.02%）'),Patch(color=ORANGE,label='Teacher 更长：8,239（62.26%）')],loc='lower center',bbox_to_anchor=(.5,.105),ncol=3,frameon=False)
fig.text(.06,.05,'注：各区间为离散分组，柱宽不代表 token 跨度；占比按 Prefix 数重新计算。',fontsize=10,color='#65748A')
save(fig,'01_length_difference_distribution.png')

fig = plt.figure(figsize=(14,10))
heading(fig,'缩短组与增长组中，正确率提升的 Prefix 均超过一半','按长度差分组展示：提升 / 不变 / 降低占比；每一行合计为 100%')
ax=fig.add_axes([.23,.2,.68,.62])
data=np.vstack([short,short.sum(0),long,long.sum(0)])
ys=np.array([0,1,2,3,4,5.3,7,8,9,10,11,12.3])
labs=labels[:5]+['缩短组合计（ΔL < 0）']+labels[6:]+['增长组合计（ΔL > 0）']
percent=data/data.sum(1)[:,None]*100
left=np.zeros(len(data))
for j,col in enumerate([GREEN,GRAY,RED]):
    ax.barh(ys,percent[:,j],left=left,color=col,height=.7)
    for y,p,l in zip(ys,percent[:,j],left):
        ax.text(l+p/2,y,f'{p:.2f}%',ha='center',va='center',fontsize=10,color='#25354B' if j==1 else 'white')
    left+=percent[:,j]
for y,n in zip(ys,data.sum(1)):
    ax.text(101.5,y,f'n={n:,}',va='center',fontsize=10,color='#65748A')
ax.set_yticks(ys,labs); ax.invert_yaxis(); ax.set_xlim(0,110)
ax.set_xticks([0,25,50,75,100],[f'{x}%' for x in [0,25,50,75,100]])
ax.tick_params(axis='y',length=0,pad=12)
ax.axvline(50,color='#52637B',lw=.8,ls=':',zorder=0)
ax.set_xlabel('该长度区间内的 Prefix 占比')
fig.legend(handles=[Patch(color=c,label=l) for c,l in zip([GREEN,GRAY,RED],['正确率提升','正确率不变','正确率降低'])],loc='lower center',bbox_to_anchor=(.55,.115),ncol=3,frameon=False)
fig.text(.06,.065,'统计口径：缩短组包含全部 ΔL < 0 的 4,993 个 Prefix；增长组包含全部 ΔL > 0 的 8,239 个 Prefix。',fontsize=10,color='#65748A')
fig.text(.06,.038,'已补入 0～+100 区间（1,954 个 Prefix）；等长组（2 个 Prefix）未提供正确率变化，未纳入。',fontsize=10,color='#65748A')
save(fig,'02_accuracy_change_shares.png')

fig=plt.figure(figsize=(14,8))
heading(fig,'已统计区间的平均正确率变化均为正','正确率变化 ΔAcc = Teacher − Student · 单位：百分点（pp）')
ax=fig.add_axes([.08,.25,.58,.56])
vals=[30.70,32.84,27.13,21.32,18.54,16.22,15.50,14.66,13.48,11.63]
x=np.array([0,1,2,3,4,6,7,8,9,10])
bars=ax.bar(x,vals,color=[BLUE]*5+[ORANGE]*5,width=.7,zorder=3)
for b,v in zip(bars,vals):
    ax.text(b.get_x()+b.get_width()/2,v+.65,f'+{v:.2f}',ha='center',fontsize=10)
ax.set_xticks(x,labels[:5]+labels[6:],rotation=32,ha='right',fontsize=9)
ax.set_ylim(0,39); ax.set_ylabel('平均正确率变化（pp）'); ax.grid(axis='y',color='#E9EDF3'); ax.set_axisbelow(True)
growth_mean = np.average(vals[5:], weights=counts[6:])
print(f'Growth mean (weighted rounded bin means): {growth_mean:.8f} pp')
ax2=fig.add_axes([.75,.30,.2,.47])
ax2.bar([0,1],[23.22,growth_mean],color=[BLUE,ORANGE],width=.6)
for i,v in enumerate([23.22,growth_mean]): ax2.text(i,v+.8,('≈ ' if i == 1 else '') + f'+{v:.2f} pp',ha='center',weight='bold',fontsize=12)
ax2.set_xticks([0,1],['缩短组合计\nΔL < 0\nn = 4,993','增长组合计\nΔL > 0\nn = 8,239'],fontsize=10)
ax2.set_ylim(0,39); ax2.set_yticks([]); ax2.set_title('合计均值',fontsize=13,pad=12)
fig.text(.06,.08,'注：增长组合计按已四舍五入的区间均值及 Prefix 数加权估算；其余均值采用提供的汇总值。',fontsize=10,color='#65748A')
fig.text(.06,.045,'分组比较描述相关性，不代表因果关系；仅凭长度分箱及占比，无法确定整体平均长度差。',fontsize=10,color='#65748A')
save(fig,'03_mean_accuracy_change.png')
print(f'Created 3 PNG files in {OUT}')
