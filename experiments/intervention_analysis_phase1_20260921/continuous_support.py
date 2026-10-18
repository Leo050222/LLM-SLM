"""Continuous paired preference and cumulative-mass position of teacher tokens."""
import json,math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from extract import OUT,save,sha
from summarize import bootstrap

DEST=OUT/'continuous_support';DEST.mkdir(exist_ok=True)
receipt=json.loads((OUT/'EXTRACTION_COMPLETE.json').read_text())
assert sha(OUT/'events.parquet')==receipt['events_sha256']
df=pd.read_parquet(OUT/'events.parquet')
assert len(df)==399601 and df.event_id.is_unique
df['D']=df.w_logpx-df.w_logpy
assert np.allclose(df.D,-df.w_log_y_over_x,atol=1e-5)
rawlo=df.w_mass_above;rawhi=rawlo+df.w_tie_mass-df.w_py
assert rawlo.min()>=-1e-6 and rawhi.max()<1+2e-6
assert ((rawhi-rawlo)>=-1e-7).all()
# Probability reductions were FP32. Clip only rounding excursions outside [0,1].
df['M_lo']=rawlo.clip(0,1);df['M_hi']=np.maximum(df.M_lo,rawhi.clip(0,1))
df['M_tie_width']=df.M_hi-df.M_lo
metrics=[]
for q in [.5,.8,.9,.95,.99,.999,.9999]:
    code=str(q).replace('.','_')
    for suffix,v in [('core_certain',df.M_hi<q),('core_possible',df.M_lo<q),
                     ('tail_certain',df.M_lo>=q),('tail_possible',df.M_hi>=q)]:
        col=f'{suffix}_{code}';df[col]=v.astype(float)
        if q in [.9,.95,.99,.999]:metrics.append(col)
    assert np.allclose(df[f'core_certain_{code}']+df[f'tail_possible_{code}'],1)
for k in [2,3,10,100]:
    df[f'near_{k}']=(df.D.abs()<=math.log(k)).astype(float)
    df[f'prefers_x_{k}']=(df.D>=math.log(k)).astype(float)
    metrics += [f'near_{k}',f'prefers_x_{k}']
df['near2_core90_certain']=((df.D.abs()<=math.log(2))&(df.M_hi<.9)).astype(float)
df['near2_core90_possible']=((df.D.abs()<=math.log(2))&(df.M_lo<.9)).astype(float)
df['prefers10_tail99_certain']=((df.D>=math.log(10))&(df.M_lo>=.99)).astype(float)
df['prefers10_tail99_possible']=((df.D>=math.log(10))&(df.M_hi>=.99)).astype(float)
metrics+=['near2_core90_certain','near2_core90_possible','prefers10_tail99_certain','prefers10_tail99_possible']
summary=[];quant=[];cis=[];macro=[]
for i,(d,g) in enumerate([('ALL',df)]+list(df.groupby('dataset'))):
    summary.append(dict(dataset=d,events=len(g),D_median=g.D.median(),M_lo_median=g.M_lo.median(),M_hi_median=g.M_hi.median(),
                        **{c:g[c].mean() for c in metrics}))
    for col in ['D','M_lo','M_hi','M_tie_width']:
        quant.append(dict(dataset=d,metric=col,**{f'p{int(q*100):02d}':v for q,v in g[col].quantile([.01,.1,.25,.5,.75,.9,.99]).items()}))
    h=g.copy();h['problem_id']=h.dataset.astype(str)+':'+h.problem_id.astype(str)
    rs,e,t=bootstrap(h,metrics,20260921+i);cis.extend(dict(dataset=d,**r) for r in rs)
    if d!='ALL':macro.append((rs,e,t))
for weight,idx in [('event',1),('question',2)]:
    arr=np.mean([v[idx] for v in macro],axis=0)
    for j,c in enumerate(metrics):
        point=np.mean([next(r['mean'] for r in v[0] if r['metric']==c and r['weight']==weight) for v in macro])
        lo,hi=np.quantile(arr[:,j],[.025,.975]);cis.append(dict(dataset='MACRO',weight=weight,metric=c,mean=point,ci_low=lo,ci_high=hi))
pd.DataFrame(summary).to_csv(DEST/'summary.csv',index=False)
pd.DataFrame(quant).to_csv(DEST/'quantiles.csv',index=False)
pd.DataFrame(cis).to_csv(DEST/'cluster_ci.csv',index=False)
grid=[]
for q in [.5,.8,.9,.95,.99,.999]:
    for k in [1.25,1.5,2,3,5,10,100]:
        grid.append(dict(mass=q,factor=k,near_core_certain=((df.D.abs()<=math.log(k))&(df.M_hi<q)).mean(),
            near_core_possible=((df.D.abs()<=math.log(k))&(df.M_lo<q)).mean(),
            prefers_x_tail_certain=((df.D>=math.log(k))&(df.M_lo>=q)).mean(),
            prefers_x_tail_possible=((df.D>=math.log(k))&(df.M_hi>=q)).mean()))
pd.DataFrame(grid).to_csv(DEST/'threshold_grid.csv',index=False)
splits=[]
df['calibration_group']=pd.cut(df[['w_replay_logpx_error','s_replay_logpx_error']].abs().max(axis=1),[-np.inf,.1,.5,1,np.inf],labels=['<=0.1','0.1-0.5','0.5-1','>1'])
for col in ['replicate','calibration_group']:
    for v,g in df.groupby(col,observed=True):splits.append(dict(split=col,value=str(v),events=len(g),D_median=g.D.median(),M_median=g.M_lo.median(),**{c:g[c].mean() for c in metrics}))
pd.DataFrame(splits).to_csv(DEST/'sensitivity.csv',index=False)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
fig,axs=plt.subplots(1,3,figsize=(17,5),layout='constrained');cdf=[]
for col,ax,label in [('D',axs[0],'D: log student preference for x over y'),('M_lo',axs[1],'Best tie position'),('M_hi',axs[1],'Worst tie position')]:
    a=np.sort(df[col].to_numpy());ix=np.unique(np.linspace(0,len(a)-1,3000).astype(int));ax.plot(a[ix],(ix+1)/len(a),label=label)
    cdf.extend(dict(metric=col,value=float(a[j]),cdf=float((j+1)/len(a))) for j in ix)
axs[0].axvline(0,ls='--',color='gray');axs[0].axvspan(-math.log(2),math.log(2),color='green',alpha=.15)
axs[0].set(xlim=(-10,35),ylim=(0,1),xlabel='D = ln p_student(x) - ln p_student(y)',ylabel='Event CDF',title='Preference gap')
axs[1].set(xlim=(0,1),ylim=(0,1),xlabel='Probability mass before teacher token',ylabel='Event CDF',title='Cumulative probability position');axs[1].legend(fontsize=8)
h=axs[2].hexbin(df.D,df.M_lo,gridsize=(65,40),extent=(-10,35,0,1),bins='log',mincnt=1,cmap='viridis')
axs[2].axvline(0,ls='--',color='red',lw=1);axs[2].set(xlim=(-10,35),ylim=(0,1),xlabel='D (positive: student prefers original token)',ylabel='M: mass strictly above teacher token',title='Joint distribution')
fig.colorbar(h,ax=axs[2],label='Events (log scale)')
fig.savefig(DEST/'continuous_distribution.png');fig.savefig(DEST/'continuous_distribution.pdf');plt.close(fig)
pd.DataFrame(cdf).to_csv(DEST/'cdf.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');curves=[]
qs=np.linspace(.01,.9999,350)
for k in [1.5,2,3,5]:
    lo=np.array([((df.D.abs()<=math.log(k))&(df.M_hi<q)).mean() for q in qs]);hi=np.array([((df.D.abs()<=math.log(k))&(df.M_lo<q)).mean() for q in qs])
    line,=axs[0].plot(qs,100*lo,label=f'Within factor {k}');axs[0].fill_between(qs,100*lo,100*hi,color=line.get_color(),alpha=.18)
    curves.extend(dict(kind='near_core',factor=k,mass=float(q),certain=float(l),possible=float(h)) for q,l,h in zip(qs,lo,hi))
for k in [3,10,100]:
    lo=np.array([((df.D>=math.log(k))&(df.M_lo>=q)).mean() for q in qs]);hi=np.array([((df.D>=math.log(k))&(df.M_hi>=q)).mean() for q in qs])
    line,=axs[1].plot(qs,100*lo,label=f'Student prefers x by >= {k}x');axs[1].fill_between(qs,100*lo,100*hi,color=line.get_color(),alpha=.18)
    curves.extend(dict(kind='prefers_x_tail',factor=k,mass=float(q),certain=float(l),possible=float(h)) for q,l,h in zip(qs,lo,hi))
axs[0].set(xlabel='Core probability mass cutoff',ylabel='% of ALL intervention events',title='Similar preference + teacher token in core')
axs[1].set(xlabel='Cumulative mass before teacher token',ylabel='% of ALL intervention events',title='Preference against teacher token + tail position')
for a in axs:a.legend(fontsize=8);a.set(xlim=(0,1),ylim=(0,None))
fig.savefig(DEST/'threshold_curves.png');fig.savefig(DEST/'threshold_curves.pdf');plt.close(fig)
pd.DataFrame(curves).to_csv(DEST/'threshold_curves.csv',index=False)
save(DEST/'COMPLETE.json',dict(events=len(df),input_sha256=receipt['events_sha256'],script_sha256=sha(__file__),
    D_definition='ln p_w(x)-ln p_w(y)',M_definition='sum probability strictly above y; upper bound adds other tied tokens',
    fp32_rounding_clipped_rows=int(((rawlo>1)|(rawhi>1)).sum()),max_raw_mass=float(rawhi.max()),
    D_plot_outside_fraction=float(((df.D< -10)|(df.D>35)).mean()),summary=summary[0]))
print(json.dumps(summary[0]),flush=True)
