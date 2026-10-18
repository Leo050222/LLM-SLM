import os
from pathlib import Path
P=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(P/'.mplcache')
import json,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
rs=json.loads((P/'all_summary.json').read_text())
styles={'Luna_abs2_k8':('Luna, threshold 2, k=8','#2563eb','-'),'Luna_abs2_k32':('Luna, threshold 2, k=32','#7c3aed','--'),'NoLuna_abs2_k8':('No Luna, threshold 2, k=8','#ea580c','-'),'NoLuna_abs3_k8':('No Luna, threshold 3, k=8','#059669','--')}
fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
for ax,(ds,title) in zip(axes.flat,[('deepmath','DeepMath (1,000)'),('math500','MATH500 (500)'),('olympiad_bench','OlympiadBench (674)'),('bbh','BBH (1,080)'),('aime24','AIME24 (30)'),('aime25','AIME25 (30)')]):
 for r in rs:
  if r['dataset']==ds:
   label,color,ls=styles[r['config']];ax.plot(range(5,100,10),r['relative_position_decile_percent'],label=label,color=color,linestyle=ls,marker='o',markersize=3,linewidth=1.8)
 ax.set_title(title);ax.set_ylim(0,20);ax.set_xlim(0,100);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
for ax in axes[-1]:ax.set_xlabel('Position in retained response (%)')
for ax in axes[:,0]:ax.set_ylabel('Share of takeover events (%)')
handles=[];labels=[]
for label,color,ls in styles.values():
 handles.append(plt.Line2D([],[],color=color,linestyle=ls,marker='o',markersize=3));labels.append(label)
fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.02),fontsize=9)
fig.suptitle('Where teacher takeovers occur',fontsize=17,y=.98)
fig.text(.5,.935,'10 equal relative-position bins; event-weighted; historical absolute-gap rules',ha='center',fontsize=10,color='#555555')
fig.tight_layout(rect=[0,.09,1,.91]);fig.savefig(P/'relative_positions.png',dpi=180);fig.savefig(P/'relative_positions.svg');print(P/'relative_positions.png')
