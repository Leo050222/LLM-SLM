"""Teacher concentration conditional on close student leading candidates."""
import json,math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from extract import OUT,save,sha
from summarize import bootstrap

DEST=OUT/'teacher_uncertainty';DEST.mkdir(exist_ok=True)
receipt=json.loads((OUT/'EXTRACTION_COMPLETE.json').read_text())
assert sha(OUT/'events.parquet')==receipt['events_sha256']
df=pd.read_parquet(OUT/'events.parquet');extra=[]
for path in sorted((OUT/'shards').glob('*.parquet')):
    assert sha(path)==json.loads(path.with_suffix('.json').read_text())['sha256']
    t=pd.read_parquet(path,columns=['event_id','w_top_logprobs','s_top_logprobs','w_masked_top_logprobs','s_masked_top_logprobs'])
    out={'event_id':t.event_id.to_numpy()}
    for tag,col in [('w','w_top_logprobs'),('s','s_top_logprobs'),('wq','w_masked_top_logprobs'),('sq','s_masked_top_logprobs')]:
        a=np.stack(t[col].to_numpy())[:,:2]
        assert (a[:,0]>=a[:,1]).all()
        out[tag+'_p1']=np.exp(a[:,0]);out[tag+'_p2']=np.exp(a[:,1])
        out[tag+'_gap']=a[:,0]-a[:,1];out[tag+'_mass2']=np.exp(a).sum(axis=1)
    extra.append(pd.DataFrame(out))
df=df.merge(pd.concat(extra,ignore_index=True),on='event_id',validate='one_to_one')
assert len(df)==399601
assert np.allclose(df.s_p1,np.exp(df.s_logpy-df.s_log_y_over_top1),rtol=1e-5,atol=1e-7)
for tag in ['w','s','sq']:
    df[tag+'_near2']=(df[tag+'_gap']<=math.log(2)).astype(float)
    df[tag+'_lead10']=(df[tag+'_gap']>=math.log(10)).astype(float)
    df[tag+'_p1ge90']=(df[tag+'_p1']>=.9).astype(float)
df['teacher_lower_entropy']=(df.w_entropy>df.s_entropy).astype(float)
df['teacher_larger_gap']=(df.s_gap>df.w_gap).astype(float)
df['teacher_greater_p1']=(df.s_p1>df.w_p1).astype(float)
df['y_in_student_top2']=(df.w_rank_min<=2).astype(float)
df['both_actual_tokens_top2']=((df.w_logpx>=np.log(df.w_p2)-1e-6)&(df.w_rank_min<=2)).astype(float)
df['s_effective_candidates']=np.exp(df.s_entropy);df['w_effective_candidates']=np.exp(df.w_entropy)
df['sq_effective_candidates']=np.exp(df.s_masked_entropy)
df['sq_entropy']=df.s_masked_entropy
groups={'ALL':np.ones(len(df),bool),'student_top2_within2':df.w_gap<=math.log(2),
        'student_top2_within2_mass80':(df.w_gap<=math.log(2))&(df.w_mass2>=.8),
        'student_top2_within2_y_core90':(df.w_gap<=math.log(2))&df.w_nucleus90_certain,
        'actual_xy_within2_y_core90':((df.w_logpx-df.w_logpy).abs()<=math.log(2))&df.w_nucleus90_certain}
measures=['w_p1','s_p1','sq_p1','w_gap','s_gap','sq_gap','w_entropy','s_entropy','sq_entropy','teacher_lower_entropy',
    'teacher_larger_gap','teacher_greater_p1','s_near2','sq_near2','s_lead10','sq_lead10','s_p1ge90','sq_p1ge90',
    'y_in_student_top2','both_actual_tokens_top2']
summary=[];quant=[];ci=[]
def describe(label,g):
    return dict(group=label,events=len(g),fraction_all=len(g)/len(df),**{c:float(g[c].mean()) for c in measures})
for label,mask in groups.items():
    g=df[mask];summary.append(describe(label,g))
    for c in ['w_p1','s_p1','sq_p1','w_p2','s_p2','sq_p2','w_gap','s_gap','sq_gap',
              'w_entropy','s_entropy','sq_entropy','w_effective_candidates','s_effective_candidates','sq_effective_candidates','w_mass2']:
        quant.append(dict(group=label,metric=c,**{f'p{int(q*100):02d}':v for q,v in g[c].quantile([.1,.25,.5,.75,.9]).items()}))
    if label in ['student_top2_within2','actual_xy_within2_y_core90']:
        h=g.copy();h['problem_id']=h.dataset.astype(str)+':'+h.problem_id.astype(str)
        rs,_,_=bootstrap(h,measures);ci.extend(dict(group=label,**r) for r in rs)
pd.DataFrame(summary).to_csv(DEST/'summary.csv',index=False)
pd.DataFrame(quant).to_csv(DEST/'quantiles.csv',index=False)
pd.DataFrame(ci).to_csv(DEST/'cluster_ci.csv',index=False)
dataset=[]
for d,g in df[groups['student_top2_within2']].groupby('dataset'):
    dataset.append(dict(dataset=d,**describe('student_top2_within2',g)))
pd.DataFrame(dataset).to_csv(DEST/'datasets.csv',index=False)
sens=[]
for factor in [1.5,2,3,5]:
    for mass in [0,.7,.8,.9]:
        g=df[(df.w_gap<=math.log(factor))&(df.w_mass2>=mass)]
        sens.append(dict(factor=factor,top2_mass_min=mass,**describe('threshold_sensitivity',g)))
pd.DataFrame(sens).to_csv(DEST/'threshold_sensitivity.csv',index=False)
splitrows=[]
df['calibration_group']=pd.cut(df[['w_replay_logpx_error','s_replay_logpx_error']].abs().max(axis=1),[-np.inf,.1,.5,1,np.inf],labels=['<=0.1','0.1-0.5','0.5-1','>1'])
for col in ['replicate','calibration_group']:
    for v,g in df[groups['student_top2_within2']].groupby(col,observed=True):
        splitrows.append(dict(split=col,value=str(v),**describe('student_top2_within2',g)))
pd.DataFrame(splitrows).to_csv(DEST/'split_sensitivity.csv',index=False)
# Literal figure regions: definitions are explicit, not inferred from visual area.
D=df.w_logpx-df.w_logpy;M=df.w_mass_above
regions={
 'lower_left_teacher_weak_top1_and_D_negative':(M==0)&(D<0),
 'D_negative':D<0,
 'upper_right_M99_Dln10':(M>=.99)&(D>=math.log(10)),
 'student_close_top2':groups['student_top2_within2']}
save(DEST/'regions.json',{k:dict(events=int(v.sum()),fraction=float(v.mean())) for k,v in regions.items()})
g=df[groups['student_top2_within2']]
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
fig,axs=plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
for tag,label,color in [('w','Student','#2475a8'),('s','Teacher raw','#c4523e'),('sq','Teacher after banning x','#329c72')]:
    for ax,metric in zip(axs,['p1','gap','entropy']):
        a=np.sort(g[tag+'_'+metric].to_numpy());ix=np.unique(np.linspace(0,len(a)-1,2500).astype(int));ax.plot(a[ix],(ix+1)/len(a),label=label,color=color)
axs[0].set(xlabel='Probability of most likely token',ylabel='Conditional event CDF',xlim=(0,1),ylim=(0,1),title='Top-1 concentration');axs[0].legend(fontsize=9)
axs[1].axvline(math.log(2),ls='--',color='gray');axs[1].set(xlabel='ln [p(top1) / p(top2)]',ylabel='Conditional event CDF',xlim=(0,15),ylim=(0,1),title='Leading candidate gap')
axs[2].set(xlabel='Full-vocabulary entropy (nats)',ylabel='Conditional event CDF',xlim=(0,5),ylim=(0,1),title='Distribution uncertainty')
fig.suptitle(f'Student top-2 probabilities within factor 2: {len(g):,} intervention events',fontsize=13)
fig.savefig(DEST/'teacher_certainty.png');fig.savefig(DEST/'teacher_certainty.pdf');plt.close(fig)
df[['event_id','dataset','problem_id','replicate','w_p1','w_p2','s_p1','s_p2','sq_p1','sq_p2','w_gap','s_gap','sq_gap','w_entropy','s_entropy','sq_entropy']].to_parquet(DEST/'top_candidates.parquet',index=False)
save(DEST/'COMPLETE.json',dict(events=len(df),input_sha256=receipt['events_sha256'],script_sha256=sha(__file__),
    summary=summary,plots='Conditional CDF; gap displayed [0,15], entropy [0,5]; all events retained in statistics',
    clipping=dict(teacher_raw_gap_above15=float((g.s_gap>15).mean()),teacher_masked_gap_above15=float((g.sq_gap>15).mean()),teacher_raw_entropy_above5=float((g.s_entropy>5).mean()))))
print(json.dumps(summary[1]),flush=True)
