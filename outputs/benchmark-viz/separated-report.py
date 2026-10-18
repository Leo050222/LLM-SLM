import os,json
from pathlib import Path
os.environ['MPLCONFIGDIR']='/private/tmp/benchmark-viz-mpl'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
out=Path(__file__).parent
data=json.loads((out/'data.json').read_text())
ks=[1,4,8,16,32]; ts=[2,2.5,3,3.5,4]
colors=['#007c91','#d77820','#7956a3','#22936e','#cd5067']
rows=sorted(data['rows'][2:],key=lambda r:(r['k'],r['t']))
ys=[ks.index(r['k'])*6+ts.index(r['t']) for r in rows]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42})
with PdfPages(out/'benchmark-separated.pdf') as pdf:
 for j,b in enumerate(data['bench']):
  fig,axs=plt.subplots(1,2,figsize=(12,13),sharey=True,gridspec_kw={'wspace':.28})
  for ax,metric,title in zip(axs,[0,2],['Accuracy (%)','Teacher output-token share (%)']):
   vals=[r['v'][j][metric] for r in rows];span=max(max(vals)-min(vals),.1)
   ax.set_xlim(max(0,min(vals)-.12*span),max(vals)+.18*span)
   ax.set_ylim(29,-1)
   ax.set_yticks(ys,[f"k={r['k']:<2}   t={r['t']:g}" for r in rows]);ax.tick_params(axis='y',length=0)
   ax.set_title(title,loc='left',pad=18,fontsize=15)
   ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
   for side in ['left','right','top']:ax.spines[side].set_visible(False)
   ax.spines['bottom'].set_color('#cbd5e1')
   for r,y,v in zip(rows,ys,vals):
    ax.scatter(v,y,s=42,c=colors[ts.index(r['t'])],zorder=3)
    ax.annotate(f'{v:.2f}%',(v,y),xytext=(8,0),textcoords='offset points',va='center',fontsize=10)
   for y in [4.8,10.8,16.8,22.8]:ax.axhline(y,color='#cbd5e1',linewidth=.8)
  axs[1].tick_params(labelleft=True)
  fig.suptitle(b,x=.08,y=.975,ha='left',fontsize=24)
  s,t=[r['v'][j][0] for r in data['rows'][:2]]
  fig.text(.08,.935,f'Baseline accuracy: Student {s:.2f}% | Teacher {t:.2f}% (teacher token shares: 0% / 100%)',fontsize=11,color='#475569')
  fig.subplots_adjust(left=.13,right=.96,top=.88,bottom=.08)
  fig.text(.08,.028,'One threshold per row; aligned panels share the same configuration ordering. Axes zoom to hybrid results.\nStopping rules differ. Token share excludes prompts / discarded drafts and does not measure total compute cost.',fontsize=10,color='#475569')
  pdf.savefig(fig)
  fig.savefig(out/f'separated-{b.replace("+","plus")}.png',dpi=150,bbox_inches='tight')
  plt.close(fig)
print(out/'benchmark-separated.pdf')
