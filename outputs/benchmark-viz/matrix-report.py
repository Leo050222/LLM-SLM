import os,json
from pathlib import Path
os.environ['MPLCONFIGDIR']='/private/tmp/benchmark-viz-mpl'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
out=Path(__file__).parent
data=json.loads((out/'data.json').read_text())
ks=[1,4,8,16,32];ts=[2,2.5,3,3.5,4];rows=data['rows'][2:]
lookup={(r['t'],r['k']):r for r in rows}
assert len(lookup)==25
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
configs=[([0,2],['Accuracy (%)','Teacher token share (%)'],['YlGnBu','YlOrBr'],'accuracy-teacher-matrix'),([1,3],['Mean interventions','Mean output tokens'],['PuBu','BuGn'],'interventions-length-matrix')]
with PdfPages(out/'benchmark-matrix-report.pdf') as pdf:
 for metrics,titles,cmaps,slug in configs:
  fig=plt.figure(figsize=(17.5,16))
  outer=fig.add_gridspec(4,2,left=.045,right=.985,bottom=.065,top=.88,hspace=.65,wspace=.18)
  for j,b in enumerate(data['bench']):
   cell=outer[j//2,j%2];pos=cell.get_position(fig)
   fig.text(pos.x0,pos.y1+.061,b,fontsize=15,weight='bold',ha='left')
   if metrics[0]==0:base=f"Baseline accuracy: Student {data['rows'][0]['v'][j][0]:.2f}% | Teacher {data['rows'][1]['v'][j][0]:.2f}%"
   else:base=f"Baseline mean output tokens: Student {data['rows'][0]['v'][j][3]:.2f} | Teacher {data['rows'][1]['v'][j][3]:.2f}; interventions 0"
   fig.text(pos.x0,pos.y1+.041,base,fontsize=9,color='#475569')
   pair=cell.subgridspec(1,2,wspace=.20)
   for m,(metric,title,cmap) in enumerate(zip(metrics,titles,cmaps)):
    ax=fig.add_subplot(pair[0,m]);a=np.array([[lookup[t,k]['v'][j][metric] for k in ks] for t in ts])
    vmin,vmax=(0,100) if metric==2 else (a.min(),a.max())
    im=ax.imshow(a,cmap=cmap,vmin=vmin,vmax=vmax,aspect='auto')
    norm=Normalize(vmin,vmax);cm=plt.get_cmap(cmap)
    for ri in range(5):
     for ci in range(5):
      rgb=cm(norm(a[ri,ci]))[:3];lum=sum(c*v for c,v in zip(rgb,[.2126,.7152,.0722]));best=metric==0 and a[ri,ci]==a.max()
      ax.text(ci,ri,f'{a[ri,ci]:.2f}',ha='center',va='center',fontsize=10,color='white' if lum<.49 else '#142c38',weight='bold' if best else 'normal')
    ax.set_xticks(range(5),ks);ax.xaxis.tick_top();ax.set_yticks(range(5),[f'{t:g}' for t in ts]);ax.tick_params(length=0,pad=4)
    ax.set_title(title+'  |  k →',fontsize=10,pad=5);ax.set_ylabel('Threshold',fontsize=9)
    ax.set_xticks(np.arange(-.5,5,1),minor=True);ax.set_yticks(np.arange(-.5,5,1),minor=True);ax.grid(which='minor',color='white',linewidth=1);ax.tick_params(which='minor',length=0)
    for sp in ax.spines.values():sp.set_visible(False)
    ax.text(0,-.055,f'Color scale: {vmin:.2f} – {vmax:.2f}',transform=ax.transAxes,fontsize=8,color='#475569',va='top')
  fig.suptitle('Threshold × k | '+('Accuracy and teacher token share' if metrics[0]==0 else 'Interventions and output length'),x=.045,ha='left',y=.994,fontsize=24)
  fig.text(.045,.964,'Qwen3-1.7B student / Qwen3-14B teacher · 8 benchmarks × 25 configurations × 2 metrics = 400 values',fontsize=11,color='#475569')
  fig.text(.045,.022,'Accuracy, interventions and length: independently scaled per benchmark. Teacher share: shared 0–100% scale. Darker = larger. Bold = best hybrid accuracy.\nTeacher share excludes prompts / discarded drafts; it is not compute cost. Stopping rules and baseline generation differ. Full source: source.md.',fontsize=9,color='#475569')
  pdf.savefig(fig);fig.savefig(out/f'{slug}.png',dpi=220,bbox_inches='tight');fig.savefig(out/f'{slug}.pdf',bbox_inches='tight');plt.close(fig)
fragment=(out/'matrix-template.html').read_text().replace('DATA_PLACEHOLDER',json.dumps(data,ensure_ascii=False))
target=out/'benchmark-matrix.html'
target.write_text(fragment)
print(json.dumps({'hybrid_configs':len(rows),'main_values':8*25*2,'supplement_values':8*25*2,'fragment':str(target)}))
