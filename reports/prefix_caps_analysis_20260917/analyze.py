import os,csv,json,collections,hashlib
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/opd_matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).parent
rows=list(csv.DictReader((P/'paired_scores.csv').open()));assert len(rows)==70400
names=sorted({r['dataset'] for r in rows});ids={ds:sorted({int(r['sample_index']) for r in rows if r['dataset']==ds}) for ds in names};assert all(len(v)==100 for v in ids.values())
A=np.full((8,100,11,8),np.nan);T=A.copy();seen=set()
for r in rows:
 key=(r['dataset'],int(r['sample_index']),int(r['cap']),int(r['replicate']));assert key not in seen;seen.add(key)
 ds,i,c,s=names.index(key[0]),ids[key[0]].index(key[1]),key[2],key[3];A[ds,i,c,s]=int(r['correct']);T[ds,i,c,s]=int(r['actual_takeovers'])
assert np.isfinite(A).all() and np.isfinite(T).all()
rng=np.random.default_rng(20260917);B=10000;cluster=A.mean(axis=3);boot_all=np.zeros((B,11));result={'datasets':{},'bootstrap':{'resamples':B,'unit':'prefix (100 per dataset), retaining all 8 replicates and all caps; overall stratified by dataset','seed':20260917}}
for d,ds in enumerate(names):
 weights=rng.multinomial(100,np.full(100,.01),size=B)/100
 boot=np.einsum('bi,ic->bc',weights,cluster[d],optimize=False);assert np.isfinite(boot).all();boot_all+=boot/8
 cells=[]
 for c in range(11):
  delta=boot[:,c]-boot[:,0];x=A[d,:,c,:];base=A[d,:,0,:];diff=cluster[d,:,c]-cluster[d,:,0]
  cells.append({'cap':c,'correct':int(x.sum()),'n':800,'accuracy_pct':float(x.mean()*100),'replicate_sd_pp':float(x.mean(axis=0).std(ddof=1)*100),'delta_pp':float((x.mean()-base.mean())*100),'relative_gain_pct':float((x.mean()/base.mean()-1)*100),'paired_delta_95ci_pp':(np.quantile(delta,[.025,.975])*100).tolist(),'mean_actual_takeovers':float(T[d,:,c,:].mean()),'prefix_any_of_8_correct_pct':float(x.any(axis=1).mean()*100),'wrong_to_right':int(((base==0)&(x==1)).sum()),'right_to_wrong':int(((base==1)&(x==0)).sum()),'improved_prefixes':int((diff>0).sum()),'worsened_prefixes':int((diff<0).sum())})
 result['datasets'][ds]=cells
result['overall']=[]
for c in range(11):
 x=A[:,:,c,:];base=A[:,:,0,:];delta=boot_all[:,c]-boot_all[:,0]
 result['overall'].append({'cap':c,'correct':int(x.sum()),'n':6400,'accuracy_pct':float(x.mean()*100),'replicate_sd_pp':float(x.mean(axis=(0,1)).std(ddof=1)*100),'delta_pp':float((x.mean()-base.mean())*100),'relative_gain_pct':float((x.mean()/base.mean()-1)*100),'paired_delta_95ci_pp':(np.quantile(delta,[.025,.975])*100).tolist(),'mean_actual_takeovers':float(T[:,:,c,:].mean()),'prefix_any_of_8_correct_pct':float(x.any(axis=2).mean()*100),'wrong_to_right':int(((base==0)&(x==1)).sum()),'right_to_wrong':int(((base==1)&(x==0)).sum())})
(P/'analysis.json').write_text(json.dumps(result,indent=2))
with (P/'accuracy_all_caps.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['dataset']+list(range(11)))
 for ds in names:w.writerow([ds]+[round(x['accuracy_pct'],3) for x in result['datasets'][ds]])
 w.writerow(['overall_equal_dataset_weight']+[round(x['accuracy_pct'],3) for x in result['overall']])
labels={'bbh':'BBH','cruxeval_i':'CRUXEval-I','cruxeval_o':'CRUXEval-O','humaneval_plus':'HumanEval+','livecodebench':'LiveCodeBench','math500':'MATH500','mbpp_plus':'MBPP+','olympiad_bench':'OlympiadBench'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(15,4.4),layout='constrained');caps=np.arange(11)
y=np.array([x['delta_pp'] for x in result['overall']]);ci=np.array([x['paired_delta_95ci_pp'] for x in result['overall']])
axs[0].plot(caps,y,'o-',color='#2459A6',linewidth=2);axs[0].fill_between(caps,ci[:,0],ci[:,1],color='#2459A6',alpha=.16);axs[0].set_title('Overall gain vs student-only');axs[0].set_ylabel('Accuracy change (percentage points)');axs[0].axhline(0,color='gray',lw=.7)
for ax,group,title in [(axs[1],['cruxeval_i','cruxeval_o','humaneval_plus','mbpp_plus','livecodebench'],'Code datasets'),(axs[2],['bbh','math500','olympiad_bench'],'Math and reasoning')]:
 for ds in group:ax.plot(caps,[x['accuracy_pct'] for x in result['datasets'][ds]],'o-',label=labels[ds],markersize=3,linewidth=1.6)
 ax.set_title(title);ax.set_ylabel('Accuracy (%)');ax.legend(fontsize=8,frameon=False)
for ax in axs:ax.set_xlabel('Maximum teacher interventions (k = 1)');ax.set_xticks(caps);ax.grid(alpha=.18)
fig.suptitle('Same frozen prefix: 100 problems/dataset × 8 paired continuations/cap',fontsize=13)
fig.savefig(P/'accuracy_curves.png',dpi=180);fig.savefig(P/'accuracy_curves.pdf');plt.close(fig)
print('OVERALL')
for r in result['overall']:print(r)
print('DATASET_CAP10')
for ds in names:print(ds,result['datasets'][ds][10])
print('CRUX_I_CAP1',result['datasets']['cruxeval_i'][1])
