"""Paired probabilities of the historical student/teacher tokens at the same prefix."""
import json,math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from extract import OUT,save,sha
from summarize import bootstrap

DEST=OUT/'probability_pair'
DEST.mkdir(exist_ok=True)
receipt=json.loads((OUT/'EXTRACTION_COMPLETE.json').read_text())
assert sha(OUT/'events.parquet')==receipt['events_sha256']
df=pd.read_parquet(OUT/'events.parquet')
assert len(df)==43879 and df.event_id.is_unique
df['delta']=df.w_py-df.w_px
df['abs_delta']=df.delta.abs()
df['log_ratio']=df.w_logpy-df.w_logpx
df['log10_ratio']=df.log_ratio/math.log(10)
df['teacher_higher']=(df.log_ratio>0).astype(float)
df['tie']=(df.log_ratio==0).astype(float)
df['within_factor2']=(df.log_ratio.abs()<=math.log(2)).astype(float)
df['within_factor10']=(df.log_ratio.abs()<=math.log(10)).astype(float)
df['teacher_lt_tenth']=(df.log_ratio<math.log(.1)).astype(float)
df['teacher_lt_hundredth']=(df.log_ratio<math.log(.01)).astype(float)
df['abs_gap_le1pp']=(df.abs_delta<=.01).astype(float)
df['abs_gap_le5pp']=(df.abs_delta<=.05).astype(float)
df['abs_gap_le10pp']=(df.abs_delta<=.1).astype(float)
df['close_both']=((df.abs_delta<=.05)&(df.log_ratio.abs()<=math.log(2))).astype(float)
df['student_at_least90_teacher_below1']=((df.w_px>=.9)&(df.w_py<.01)).astype(float)
cols=['w_px','w_py','delta','abs_delta','teacher_higher','tie','within_factor2','within_factor10',
      'teacher_lt_tenth','teacher_lt_hundredth','abs_gap_le1pp','abs_gap_le5pp','abs_gap_le10pp','close_both',
      'student_at_least90_teacher_below1']
rows=[];quant=[];ci=[]
for d,g in [('ALL',df)]+list(df.groupby('dataset')):
    r=dict(dataset=d,events=len(g),**{c:float(g[c].mean()) for c in cols},
           p_student_median=g.w_px.median(),p_teacher_median=g.w_py.median(),
           paired_delta_median=g.delta.median(),ratio_median=float(np.exp(g.log_ratio.median())))
    rows.append(r)
    for c in ['w_px','w_py','delta','abs_delta','log10_ratio','s_py','s_qy']:
        quant.append(dict(dataset=d,metric=c,**{f'p{int(q*100):02d}':v for q,v in g[c].quantile([.01,.1,.25,.5,.75,.9,.99]).items()}))
    h=g.copy();h['problem_id']=h.dataset.astype(str)+':'+h.problem_id.astype(str)
    b,_,_=bootstrap(h,cols);ci.extend(dict(dataset=d,**v) for v in b)
pd.DataFrame(rows).to_csv(DEST/'paired_summary.csv',index=False)
pd.DataFrame(quant).to_csv(DEST/'paired_quantiles.csv',index=False)
pd.DataFrame(ci).to_csv(DEST/'cluster_ci.csv',index=False)
edges=[-np.inf,-3,-2,-1,math.log10(.5),0,math.log10(2),1,np.inf]
labels=['<0.001','0.001-0.01','0.01-0.1','0.1-0.5','0.5-1','1-2','2-10','>=10']
df['ratio_bin']=pd.cut(df.log10_ratio,edges,labels=labels,right=False)
df.groupby(['dataset','ratio_bin'],observed=False).size().rename('events').to_csv(DEST/'ratio_bins.csv')
df.groupby('ratio_bin',observed=False).size().rename('events').to_csv(DEST/'ratio_bins_all.csv')
splits=[]
df['calibration_group']=pd.cut(df[['w_replay_logpx_error','s_replay_logpx_error']].abs().max(axis=1),[-np.inf,.1,.5,1,np.inf],labels=['<=0.1','0.1-0.5','0.5-1','>1'])
for col in ['replicate','calibration_group','support_group']:
    for v,g in df.groupby(col,observed=True):splits.append(dict(split=col,value=str(v),events=len(g),**{c:g[c].mean() for c in cols}))
pd.DataFrame(splits).to_csv(DEST/'sensitivity.csv',index=False)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained')
cdf=[]
def plotcdf(ax,col,label,scale=1):
    vals=np.sort(df[col].to_numpy());ix=np.unique(np.linspace(0,len(vals)-1,2500).astype(int))
    ax.plot(vals[ix]*scale,(ix+1)/len(vals),label=label)
    cdf.extend(dict(metric=col,value=float(vals[i]),cdf=float((i+1)/len(vals))) for i in ix)
plotcdf(axs[0,0],'w_px','Original student token x')
plotcdf(axs[0,0],'w_py','Actual teacher token y')
axs[0,0].set(xlabel='Probability under the SAME student distribution',ylabel='Event CDF',xlim=(0,1),ylim=(0,1));axs[0,0].legend(fontsize=9)
plotcdf(axs[0,1],'delta','Paired probability difference',100)
axs[0,1].axvline(0,color='gray',ls='--');axs[0,1].set(xlabel='p_student(y) - p_student(x) [percentage points]',ylabel='Event CDF',xlim=(-100,100),ylim=(0,1))
plotcdf(axs[1,0],'log10_ratio','Paired probability ratio')
axs[1,0].axvspan(math.log10(.5),math.log10(2),alpha=.15,color='green',label='Within factor 2')
axs[1,0].axvline(0,color='gray',ls='--');axs[1,0].set(xlabel='log10 [p_student(y) / p_student(x)]',ylabel='Event CDF',xlim=(-12,3),ylim=(0,1));axs[1,0].legend(fontsize=9)
h=axs[1,1].hexbin(df.w_px,df.w_py,gridsize=55,bins='log',mincnt=1,cmap='viridis')
axs[1,1].plot([0,1],[0,1],ls='--',color='red',lw=1)
axs[1,1].set(xlabel='Student probability of original student token x',ylabel='Student probability of actual teacher token y',xlim=(0,1),ylim=(0,1))
fig.colorbar(h,ax=axs[1,1],label='Events (log scale)')
fig.savefig(DEST/'paired_probability.png');fig.savefig(DEST/'paired_probability.pdf');plt.close(fig)
pd.DataFrame(cdf).to_csv(DEST/'cdf.csv',index=False)
save(DEST/'COMPLETE.json',dict(events=len(df),input_sha256=receipt['events_sha256'],script_sha256=sha(__file__),
    interpretation='Primary comparisons use SAME unmasked student distribution; s_qy is secondary teacher sampling distribution after banning x. Replay probabilities, not original vLLM tensors.',
    plot_note='Log-ratio plot displays [-12,3]; all tails included in statistics and CSV.',summary=rows[0]))
print(json.dumps(rows[0]),flush=True)
