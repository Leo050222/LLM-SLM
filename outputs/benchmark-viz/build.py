import csv, json, os, shutil
from pathlib import Path
os.environ['MPLCONFIGDIR']='/private/tmp/benchmark-viz-mpl'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

OUT=Path(__file__).parent
SRC=OUT/'source.md'
lines=[x for x in SRC.read_text().splitlines() if x.startswith('|')]
bench=[x.strip() for x in lines[0].split('|')[4:-1]]
rows=[]
for line in lines[2:]:
    c=[x.strip() for x in line.split('|')[1:-1]]
    rows.append(dict(name=c[0],t=None if c[1]=='—' else float(c[1]),k=None if c[2]=='—' else int(c[2]),v=[[float(z.strip().replace('%','')) for z in cell.split('/')] for cell in c[3:]]))
assert len(rows)==27 and len(bench)==8
assert all(len(r['v'])==8 and all(len(v)==4 for v in r['v']) for r in rows)
data=dict(bench=bench,rows=rows)
(OUT/'data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
if SRC.resolve() != (OUT/'source.md').resolve():shutil.copyfile(SRC,OUT/'source.md')
with (OUT/'data.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['experiment','threshold','k','benchmark','accuracy_pct','interventions','teacher_token_pct','output_tokens'])
    for r in rows:
        for b,v in zip(bench,r['v']):w.writerow([r['name'],r['t'],r['k'],b,*v])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#334155','text.color':'#172b3a','axes.titleweight':'medium','pdf.fonttype':42})
colors=['#007c91','#d77820','#7956a3','#22936e','#cd5067']
ts=[2,2.5,3,3.5,4]
pdf=PdfPages(OUT/'benchmark-report.pdf')
fig,axs=plt.subplots(2,4,figsize=(18,9.5))
for j,(b,ax) in enumerate(zip(bench,axs.flat)):
    for ti,t in enumerate(ts):
        rr=[r for r in rows if r['t']==t]
        for r,marker in zip(rr,['o','^','s','D','P']):
            ax.scatter(r['v'][j][2],r['v'][j][0],c=colors[ti],s=54,marker=marker,edgecolors='white',linewidths=.4,zorder=3)
    for i,mark in [(0,'X'),(1,'*')]:
        r=rows[i];ax.scatter(r['v'][j][2],r['v'][j][0],marker=mark,s=90 if i==0 else 160,color='#172b3a',zorder=4)
        ax.axhline(r['v'][j][0],color='#64748b',lw=.7,ls='--',alpha=.5)
    pts=[r['v'][j] for r in rows]
    front=[r for r in rows if not any(s['v'][j][2]<=r['v'][j][2] and s['v'][j][0]>=r['v'][j][0] and (s['v'][j][2]<r['v'][j][2] or s['v'][j][0]>r['v'][j][0]) for s in rows)]
    front.sort(key=lambda r:r['v'][j][2])
    ax.plot([r['v'][j][2] for r in front],[r['v'][j][0] for r in front],c='#64748b',lw=.8,alpha=.6,zorder=1)
    best=max(rows[2:],key=lambda r:r['v'][j][0]);v=best['v'][j]
    ax.annotate(f"{best['name']}\n{v[0]:.2f}%",(v[2],v[0]),xytext=(5,-30),textcoords='offset points',fontsize=9,color='#172b3a')
    ax.set_title(b,loc='left',pad=12);ax.set_xlabel('Teacher output-token share (%)');ax.set_ylabel('Accuracy (%)');ax.grid(alpha=.14);ax.margins(.09,.16)
handles=[Line2D([],[],marker='o',ls='',color=c,label=f't = {t:g}') for t,c in zip(ts,colors)]
handles += [Line2D([],[],marker='X',ls='',color='#172b3a',label='Student only'),Line2D([],[],marker='*',ls='',color='#172b3a',label='Teacher only'),Line2D([],[],color='#64748b',lw=.8,label='Observed Pareto frontier')]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.95),ncol=8,frameon=False)
khandles=[Line2D([],[],marker=m,ls='',color='#172b3a',label=f'k = {k}') for k,m in zip([1,4,8,16,32],['o','^','s','D','P'])]
fig.legend(handles=khandles,loc='upper center',bbox_to_anchor=(.5,.915),ncol=5,frameon=False)
fig.suptitle('Accuracy vs. teacher token share',x=.06,ha='left',fontsize=23,y=.99)
fig.text(.06,.02,'Qwen3-1.7B → Qwen3-14B | Each point is one configuration. Axes differ by benchmark.\nToken share excludes prompts / discarded drafts; it is not latency or total compute cost. Stopping rules differ; no uncertainty estimates supplied.',fontsize=10,color='#475569')
fig.tight_layout(rect=[0,.08,1,.89]);fig.savefig(OUT/'accuracy-vs-teacher-share.png',dpi=190,bbox_inches='tight');pdf.savefig(fig);plt.close(fig)
metrics=[('Accuracy (%)','accuracy','YlGnBu'),('Mean interventions','interventions','YlOrBr'),('Teacher output-token share (%)','teacher-share','PuBu'),('Mean output tokens','output-tokens','BuGn')]
a=np.array([r['v'] for r in rows])
for mi,(title,slug,cmap) in enumerate(metrics):
    vals=a[:,:,mi];lo=vals.min(axis=0);hi=vals.max(axis=0);norm=(vals-lo)/np.where(hi>lo,hi-lo,1)
    fig,ax=plt.subplots(figsize=(13,14))
    ax.imshow(norm,cmap=cmap,vmin=0,vmax=1,aspect='auto')
    for i in range(27):
        for j in range(8):
            ax.text(j,i,f'{vals[i,j]:.2f}',ha='center',va='center',fontsize=10,color='white' if norm[i,j]>.60 else '#152b3a',weight='bold' if mi==0 and i>=2 and vals[i,j]==vals[2:,j].max() else 'normal')
    ax.set_xticks(range(8),bench,rotation=22,ha='left');ax.xaxis.tick_top();ax.tick_params(axis='both',length=0,pad=9)
    ax.set_yticks(range(27),[r['name'] for r in rows])
    for y in [1.5,6.5,11.5,16.5,21.5]:ax.axhline(y,color='white',lw=3)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.suptitle(title,x=.16,y=.975,ha='left',fontsize=23)
    fig.text(.16,.027,'Cell values are raw measurements; color is scaled independently within each benchmark column.\nDarker = larger (not always better). Bold accuracy = best hybrid configuration within that benchmark.\nStopping rules / baseline generation differ. See source.md for sample counts and full methodology.',fontsize=10,color='#475569')
    fig.tight_layout(rect=[0,.08,1,.94]);fig.savefig(OUT/f'{slug}-heatmap.png',dpi=180,bbox_inches='tight');pdf.savefig(fig);plt.close(fig)
pdf.close()
template=OUT/'interactive-template.html'
target=OUT/'benchmark-tradeoffs.html'
target.write_text(template.read_text().replace('DATA_PLACEHOLDER',json.dumps(data,ensure_ascii=False)))
print(json.dumps({'rows':len(rows),'benchmarks':len(bench),'measurements':int(a.size),'output':str(OUT),'inline':str(target)},ensure_ascii=False))
