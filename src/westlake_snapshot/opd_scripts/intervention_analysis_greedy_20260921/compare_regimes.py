"""Descriptive comparison across independent greedy/stochastic trajectories."""
import json,math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from extract import OUT,ROOT,save,sha

OLD=ROOT/'opd_analysis/intervention_analysis_phase1_20260921'
DEST=OUT/'regime_comparison';DEST.mkdir(exist_ok=True)
frames={};rows=[];sensitivity=[]
for label,path in [('Non-greedy x8',OLD),('Dual greedy',OUT)]:
    d=pd.read_parquet(path/'events.parquet');d['D']=d.w_logpx-d.w_logpy;d['M']=d.w_mass_above.clip(0,1)
    d['near_core']=((d.D.abs()<=math.log(2))&d.w_nucleus90_certain).astype(float)
    d['tail99']=((d.D>=math.log(10))&(d.M>=.99)).astype(float)
    d['entropy_lower']=(d.entropy_delta>0).astype(float)
    d['weak_argmax_regret']=(d.w_logpy-d.w_log_y_over_top1)-d.w_logpx
    d['weak_greedy_compatible']=(d.weak_argmax_regret.abs()<1e-5).astype(float)
    d['teacher_greedy_compatible']=(d.s_masked_rank_min==1).astype(float)
    frames[label]=d
    for ds,g in [('ALL',d)]+list(d.groupby('dataset')):
        rows.append(dict(regime=label,dataset=ds,events=len(g),weak_p_y_median=g.w_py.median(),weak_p_x_median=g.w_px.median(),
            weak_q_y_median=g.w_qy.median(),top5=g.w_top5_certain.mean(),top10=g.w_top10_certain.mean(),
            high=g.support_group.eq('high').mean(),low=g.support_group.eq('low').mean(),D_median=g.D.median(),M_median=g.M.median(),
            near_core=g.near_core.mean(),tail99=g.tail99.mean(),weak_entropy=g.w_entropy.mean(),strong_entropy=g.s_entropy.mean(),
            delta=g.entropy_delta.mean(),masked_delta=g.masked_entropy_delta.mean(),entropy_lower=g.entropy_lower.mean(),
            weak_greedy_compatible=g.weak_greedy_compatible.mean(),teacher_greedy_compatible=g.teacher_greedy_compatible.mean(),
            teacher_higher_in_weak=(g.D<0).mean()))
    for name,mask in [('all',np.ones(len(d),bool)),('no_timeout',d.finish_reason.ne('trace_timeout')),
                      ('small_calibration',d[['w_replay_logpx_error','s_replay_logpx_error']].abs().max(axis=1)<=.1)]:
        g=d[mask];sensitivity.append(dict(regime=label,subset=name,events=len(g),near_core=g.near_core.mean(),tail99=g.tail99.mean(),top5=g.w_top5_certain.mean()))
pd.DataFrame(rows).to_csv(DEST/'comparison.csv',index=False)
pd.DataFrame(sensitivity).to_csv(DEST/'sensitivity.csv',index=False)
d=frames['Dual greedy'];top=pd.read_parquet(OUT/'teacher_uncertainty/top_candidates.parquet')
d=d.merge(top[['event_id','sq_p1']],on='event_id',validate='one_to_one')
d['teacher_argmax_regret']=np.log(d.sq_p1)-d.s_logqy
audit=dict(events=len(d),student_original_is_replay_top1=float(d.weak_greedy_compatible.mean()),
    teacher_actual_is_replay_masked_top1=float(d.teacher_greedy_compatible.mean()),
    D_negative_events=int((d.D<0).sum()),D_zero_events=int((d.D==0).sum()),
    note='Historical T=0 actions can differ from replay argmax due to backend/BF16 drift; do not infer stochastic sampling from D<0.')
for col in ['weak_argmax_regret','teacher_argmax_regret']:
    audit[col]={str(q):float(v) for q,v in d[col].quantile([.5,.9,.95,.99,1]).items()}
save(DEST/'greedy_argmax_audit.json',audit)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
fig,axs=plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
for label,d in frames.items():
    for ax,col in zip(axs,['D','M','entropy_delta']):
        a=np.sort(d[col].to_numpy());ix=np.unique(np.linspace(0,len(a)-1,2500).astype(int));ax.plot(a[ix],(ix+1)/len(a),label=label)
axs[0].set(xlim=(-8,30),xlabel='D: ln student preference for original token',ylabel='Event CDF',title='Paired preference gap')
axs[1].set(xlim=(0,1),xlabel='M: mass above actual teacher token',ylabel='Event CDF',title='Teacher token position in student')
axs[2].set(xlim=(-2,2),xlabel='Weak minus strong entropy (nats)',ylabel='Event CDF',title='Paired entropy difference')
for a in axs:a.legend(fontsize=9);a.set(ylim=(0,1))
fig.savefig(DEST/'greedy_vs_nongreedy.png');fig.savefig(DEST/'greedy_vs_nongreedy.pdf');plt.close(fig)
save(DEST/'COMPLETE.json',dict(script_sha256=sha(__file__),comparison='Descriptive independent trajectory comparison; states and event populations differ; greedy has one rollout, stochastic eight.',
    sources={label:dict(events=len(d)) for label,d in frames.items()},argmax_audit=audit))
print(json.dumps(audit),flush=True)
