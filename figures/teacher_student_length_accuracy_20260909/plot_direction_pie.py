from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
font=FontProperties(fname='/System/Library/Fonts/Hiragino Sans GB.ttc').get_name()
plt.rcParams.update({'font.family':[font,'DejaVu Sans'],'text.color':'#263348'})
fig,ax=plt.subplots(figsize=(7,6),facecolor='white')
fig.subplots_adjust(top=.86,bottom=.16,left=.12,right=.88)
colors=['#4578B9','#BFC9D6','#D58C3E']
wedges,texts,auto=ax.pie([4993,2,8239],colors=colors,startangle=90,autopct=lambda p:f'{p:.2f}%' if p>1 else '',pctdistance=.65,textprops={'fontsize':20,'color':'white'},wedgeprops={'linewidth':0})
ax.set_title('Teacher continuation 长度分布',fontsize=19,pad=20)
fig.legend([Patch(color=c) for c in colors],['更短','相同（0.02%）','更长'],loc='lower center',bbox_to_anchor=(.5,.045),ncol=3,frameon=False,fontsize=12)
ax.set_aspect('equal')
p=Path(__file__).resolve().parent/'05_length_direction_pie.png'
fig.savefig(p,dpi=300,facecolor='white')
print(p)
