from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

out = Path(__file__).resolve().parent
font = FontProperties(fname='/System/Library/Fonts/Hiragino Sans GB.ttc').get_name()
plt.rcParams.update({'font.family':[font,'DejaVu Sans'], 'axes.unicode_minus':False, 'font.size':13, 'text.color':'#202C40', 'axes.labelcolor':'#65748A', 'xtick.color':'#65748A', 'ytick.color':'#202C40'})
counts = [4993,2,8239]
total = sum(counts)
pcts = [100*n/total for n in counts]
fig = plt.figure(figsize=(12,6.5),facecolor='white')
fig.text(.06,.93,'Teacher continuation 长度方向分布',fontsize=23,weight='bold',va='top')
fig.text(.06,.85,f'相对 Student continuation · 共 {total:,} 个 Prefix',fontsize=12,color='#65748A')
ax=fig.add_axes([.28,.25,.65,.49])
colors=['#4578B9','#BFC9D6','#D58C3E']
bars=ax.barh([0,1,2],pcts,color=colors,height=.52,zorder=3)
for y,n,p in zip([0,1,2],counts,pcts):
    ax.text(p+1.5,y,f'{n:,}  |  {p:.2f}%',va='center',ha='left',fontsize=15,weight='bold')
ax.set_yticks([0,1,2],['Teacher continuation 更短','完全相同','Teacher continuation 更长'],fontsize=12)
ax.invert_yaxis()
ax.set_xlim(0,84)
ax.set_xticks([0,20,40,60,80],['0%','20%','40%','60%','80%'])
ax.grid(axis='x',color='#E9EDF3',zorder=0)
ax.tick_params(axis='both',length=0,pad=12)
for spine in ax.spines.values(): spine.set_visible(False)
ax.set_xlabel('Prefix 占比',labelpad=14)
fig.text(.06,.07,'注：占比按 Prefix 数重新计算；更短为 37.73%（原表为 37.74%）。百分比四舍五入后合计可能不等于 100%。',fontsize=10,color='#65748A')
path=out/'04_length_direction_summary.png'
fig.savefig(path,dpi=300,facecolor='white')
plt.close(fig)
print(path)
