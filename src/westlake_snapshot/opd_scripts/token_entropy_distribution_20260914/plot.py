"""Publication-exportable distributions from audited histogram counts."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
groups={}
with (a.out/'histograms.csv').open() as f:
    for r in csv.DictReader(f):
        if r['config']=='ALL' and r['token_source']=='all':
            groups.setdefault(r['dataset'],[]).append(r)
colors={'student':'#1768AC','teacher':'#D66526'}
def series(ds,role,weight):
    rows=sorted(groups[ds],key=lambda r:float(r['left']))
    x=np.array([(float(r['left'])+float(r['right']))/2 for r in rows])
    key=role+('_count' if weight=='token' else '_trace_equal_mass')
    y=np.array([float(r[key]) for r in rows]);y/=y.sum()
    return x,y

plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for j,weight in enumerate(['token','trace']):
    for role in colors:
        x,y=series('ALL',role,weight)
        axs[j,0].stairs(np.maximum(y/.02,1e-10),np.arange(641)*.02,label=role.title(),color=colors[role])
        axs[j,1].plot(x+.01,y.cumsum(),label=role.title(),color=colors[role])
    axs[j,0].set(yscale='log',ylim=(1e-6,100),xlim=(0,12.8),ylabel='Density (log scale)',
                 title='Token-weighted' if weight=='token' else 'Equal weight per nonempty trace')
    axs[j,1].set(ylim=(0,1.01),xlim=(0,8),ylabel='Cumulative probability',title='CDF (zoom to 8 nats)')
    for ax in axs[j]:ax.set_xlabel('Full-vocabulary entropy (nats)');ax.legend();ax.grid(alpha=.15)
fig.suptitle('Student 1.7B vs Teacher 14B: identical response prefixes\n8 datasets × 25 configurations; descriptive mixture')
fig.savefig(a.out/'overall_entropy_distribution.png',dpi=180)
fig.savefig(a.out/'overall_entropy_distribution.pdf');plt.close(fig)
datasets=[d for d in groups if d!='ALL']
fig,axs=plt.subplots(2,4,figsize=(15,7),layout='constrained')
for ax,ds in zip(axs.flat,datasets):
    for role in colors:
        x,y=series(ds,role,'token');ax.plot(x+.01,y.cumsum(),color=colors[role],label=role.title())
    ax.set(title=ds,xlim=(0,6),ylim=(0,1.01),xlabel='Entropy (nats)',ylabel='CDF');ax.grid(alpha=.15)
axs.flat[0].legend();fig.suptitle('Full-token entropy CDF by dataset; all 25 configurations')
fig.savefig(a.out/'dataset_entropy_cdf.png',dpi=180)
fig.savefig(a.out/'dataset_entropy_cdf.pdf')
print('PLOTS_COMPLETE',flush=True)
