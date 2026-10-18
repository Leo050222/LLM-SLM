import hashlib,json,math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from extract import OUT,ARCHIVE,save,sha,EXPECTED_EVENTS,EXPECTED_TRACES

def bootstrap(g,cols,seed=20260921):
    # Resample whole questions, retaining all correlated repeats and events.
    q=g.groupby('problem_id')[cols].agg(['sum','count'])
    r=g.groupby(['problem_id','replicate'])[cols].mean().groupby('problem_id').mean().reindex(q.index)
    rng=np.random.default_rng(seed);n=len(q)
    ev=[];eq=[]
    sums=q.xs('sum',axis=1,level=1).to_numpy(float);counts=q.xs('count',axis=1,level=1).to_numpy(float)
    vals=r.to_numpy(float)
    for start in range(0,2000,100):
        ix=rng.integers(0,n,size=(min(100,2000-start),n))
        ev.append(sums[ix].sum(1)/counts[ix].sum(1));eq.append(vals[ix].mean(1))
    e=np.concatenate(ev);t=np.concatenate(eq)
    rows=[]
    for j,c in enumerate(cols):
        for weight,v,point in [('event',e[:,j],g[c].mean()),('question',t[:,j],r[c].mean())]:
            lo,hi=np.quantile(v,[.025,.975]);rows.append(dict(metric=c,weight=weight,mean=point,ci_low=lo,ci_high=hi))
    return rows,e,t

def basic(g):
    return dict(events=len(g),questions=len(g[['dataset','problem_id']].drop_duplicates()),
        high=g.high.mean(),middle=g.middle.mean(),low=g.low.mean(),
        p_median=g.w_py.median(),q_median=g.w_qy.median(),rank_median=g.w_rank_min.median(),
        top5=g.w_top5_certain.mean(),top10=g.w_top10_certain.mean(),top50=g.w_top50_certain.mean(),
        weak_entropy=g.w_entropy.mean(),strong_entropy=g.s_entropy.mean(),delta=g.entropy_delta.mean(),
        weak_higher=g.weak_higher.mean(),masked_delta=g.masked_entropy_delta.mean(),
        weak_masked_entropy=g.w_masked_entropy.mean(),strong_masked_entropy=g.s_masked_entropy.mean(),
        gap_flip=g.gap_flip.mean())

def main():
    receipt=json.loads((OUT/'EXTRACTION_COMPLETE.json').read_text())
    assert sha(OUT/'events.parquet')==receipt['events_sha256']
    assert sha(OUT/'traces.parquet')==receipt['traces_sha256']
    df=pd.read_parquet(OUT/'events.parquet');tr=pd.read_parquet(OUT/'traces.parquet')
    assert len(df)==EXPECTED_EVENTS and len(tr)==EXPECTED_TRACES and df.event_id.is_unique
    assert sorted(df.event_id)==list(range(EXPECTED_EVENTS))
    assert not df.duplicated(['dataset','problem_id','replicate','event_index']).any()
    assert tr.interventions.sum()==EXPECTED_EVENTS
    for c in ['high','middle','low']:df[c]=df.support_group.eq(c).astype(float)
    df['weak_higher']=(df.entropy_delta>0).astype(float)
    df['gap_flip']=(df.replay_gap<=2).astype(float)
    df['p_lt001']=(df.w_py<.001).astype(float)
    df['p_ge01']=(df.w_py>=.01).astype(float)
    df['p_ge10']=(df.w_py>=.1).astype(float)
    df['low_p_high_q']=((df.w_py<.001)&(df.w_qy>=.1)).astype(float)
    df['generation_timeout']=df.finish_reason.eq('trace_timeout')
    score_issues=['scoring_execution_timeout','solution_process_crash','extraction_missing']
    df['score_issue']=df.score_status.isin(score_issues)
    df['first_intervention']=df.intervention_ordinal.eq(0)
    df['round_group']='greedy_single'
    df['calibration_max_error']=df[['w_replay_logpx_error','s_replay_logpx_error']].abs().max(axis=1)
    df['calibration_group']=pd.cut(df.calibration_max_error,[-np.inf,.1,.5,1,np.inf],labels=['<=0.1','0.1-0.5','0.5-1','>1']).astype(str)
    df['boundary']=((df.w_rank_min<=5)&(df.w_rank_max>5))|((df.w_rank_min<=50)&(df.w_rank_max>50))
    for t in [.001,.01]:df['boundary']|=df.w_py.between(t*.9,t*1.1)
    cols=['high','middle','low','w_py','w_qy','p_lt001','p_ge01','p_ge10','low_p_high_q',
          'w_top1_certain','w_top5_certain','w_top10_certain','w_top50_certain',
          'w_top1_possible','w_top5_possible','w_top10_possible','w_top50_possible',
          'w_nucleus90_certain','w_nucleus90_possible','w_nucleus95_certain','w_nucleus95_possible',
          'w_entropy','s_entropy','w_masked_entropy','s_masked_entropy','entropy_delta','masked_entropy_delta','weak_higher','gap_flip']
    ds=list(df.dataset.unique());summary=[];intervals=[];macroe=[];macroq=[]
    for i,(d,g) in enumerate(df.groupby('dataset',sort=True)):
        summary.append(dict(dataset=d,weight='event',**basic(g)))
        rs,e,q=bootstrap(g,cols,20260921+i);intervals.extend(dict(dataset=d,**r) for r in rs)
        macroe.append(e);macroq.append(q)
    ci=pd.DataFrame(intervals)
    for weight,arr in [('event_dataset_macro',np.mean(macroe,axis=0)),('question_dataset_macro',np.mean(macroq,axis=0))]:
        for j,c in enumerate(cols):
            point=ci[(ci.metric==c)&(ci.weight==('event' if weight.startswith('event') else 'question'))]['mean'].mean()
            lo,hi=np.quantile(arr[:,j],[.025,.975]);intervals.append(dict(dataset='MACRO',metric=c,weight=weight,mean=point,ci_low=lo,ci_high=hi))
    pooled_input=df.copy()
    pooled_input['problem_id']=pooled_input.dataset.astype(str)+':'+pooled_input.problem_id.astype(str)
    pooled_rows,_,_=bootstrap(pooled_input,cols)
    intervals.extend(dict(dataset='ALL_EVENT_POOL',**r) for r in pooled_rows)
    pd.DataFrame(intervals).to_csv(OUT/'cluster_bootstrap_summary.csv',index=False)
    pd.DataFrame(summary).to_csv(OUT/'support_entropy_summary.csv',index=False)
    quant=[]
    for d,g in [('ALL_EVENT_POOL',df)]+list(df.groupby('dataset')):
        for c in ['w_py','w_qy','w_rank_min','w_rank_max','w_entropy','s_entropy','entropy_delta','masked_entropy_delta']:
            q=g[c].quantile([.01,.1,.25,.5,.75,.9,.99]);quant.append(dict(dataset=d,metric=c,**{f'p{int(k*100)}':v for k,v in q.items()}))
    pd.DataFrame(quant).to_csv(OUT/'quantiles.csv',index=False)
    splits=[]
    for col in ['support_group','token_class','replicate','round_group','first_intervention','correct','generation_timeout','score_issue','calibration_group','boundary']:
        for (d,v),g in df.groupby(['dataset',col],dropna=False,observed=True):splits.append(dict(dataset=d,split=col,value=str(v),**basic(g)))
    pd.DataFrame(splits).to_csv(OUT/'stratified_summary.csv',index=False)
    sensitivity=[]
    for d,g in df.groupby('dataset'):
        for k in [5,10,50]:
            for p in [.001,.01,.05]:
                v=(g.w_rank_max<=k)&(g.w_py>=p)
                q=g[['problem_id','replicate']].assign(value=v).groupby(['problem_id','replicate']).value.mean().groupby('problem_id').mean().mean()
                sensitivity.append(dict(dataset=d,rank_cutoff=k,probability_cutoff=p,event_fraction=v.mean(),question_fraction=q))
    pd.DataFrame(sensitivity).to_csv(OUT/'threshold_sensitivity.csv',index=False)
    robustness=[]
    for d,g in [('ALL_EVENT_POOL',df)]+list(df.groupby('dataset')):
        for label,h in [('all',g),('exclude_boundary',g[~g.boundary]),('calibration_le_01',g[g.calibration_max_error<=.1]),
                        ('calibration_gt_1',g[g.calibration_max_error>1]),('exclude_timeout',g[~g.generation_timeout]),
                        ('exclude_score_issue',g[~g.score_issue]),
                        ('exclude_EOS_whitespace_punctuation',g[~g.token_class.isin(['EOS','whitespace','punctuation_symbol'])])]:
            if len(h):robustness.append(dict(dataset=d,subset=label,**basic(h)))
    pd.DataFrame(robustness).to_csv(OUT/'robustness.csv',index=False)
    df['probability_bin']=pd.cut(df.w_py,[-np.inf,.001,.01,.1,np.inf],right=False,labels=['<0.1%','0.1-1%','1-10%','>=10%'])
    df.groupby(['dataset','probability_bin'],observed=False).size().rename('events').to_csv(OUT/'probability_bins.csv')
    tr.groupby(['dataset','score_status'],dropna=False).size().rename('traces').to_csv(OUT/'score_status.csv')
    coverage=tr.groupby('dataset').agg(traces=('source','size'),questions=('problem_id','nunique'),events=('interventions','sum'),
                no_intervention=('interventions',lambda x:int((x==0).sum())),missing_scores=('correct',lambda x:int(x.isna().sum())))
    coverage.to_csv(OUT/'coverage.csv')
    tr[tr.correct.isna()].to_csv(OUT/'missing_scores.csv',index=False)
    tr[tr.score_status.isin(score_issues)].to_csv(OUT/'score_issue_traces.csv',index=False)
    audit=[]
    for d,g in [('ALL_EVENT_POOL',df)]+list(df.groupby('dataset')):
        r=dict(dataset=d,events=len(g),gap_flip=g.gap_flip.mean(),ties_fraction=(g.w_tie_count>1).mean(),boundary_fraction=g.boundary.mean(),
               teacher_top16_overlap=g.s_historical_top16_overlap.mean(),teacher_top16_tie_compatible=g.s_historical_top16_at_or_above_cutoff.mean())
        for c in ['w_replay_logpx_error','s_replay_logpx_error','s_historical_top16_abs_error_mean','s_historical_top16_abs_error_max']:
            r[c+'_signed_mean']=g[c].mean()
            for q in [.5,.9,.95,.99,1.]:r[c+f'_abs_p{int(q*100)}']=g[c].abs().quantile(q)
        r['historical_top16_available_events']=int((g.s_historical_top16_count>0).sum())
        r={k:(None if isinstance(v,float) and not math.isfinite(v) else v) for k,v in r.items()}
        audit.append(r)
    save(OUT/'measurement_audit.json',dict(coverage_valid=True,events=EXPECTED_EVENTS,missing_scores=int(tr.correct.isna().sum()),
           calibration=audit,warning='Replay-only measurements. No historical weak logits for actual teacher token; cannot directly validate its rank drift. Missing historical teacher top16 entries are null, not zero.'))
    # Three compact diagnostic figures; paired entropy uses event weighting explicitly.
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,ax=plt.subplots(2,4,figsize=(16,7),layout='constrained')
    cdfrows=[]
    for a,d in zip(ax.flat,ds):
        g=df[df.dataset==d]
        for col,label,color in [('w_py','Original weak probability','#186e9d'),('w_qy','After excluding original token','#be4a3d')]:
            vals=np.sort(g[col].to_numpy());ix=np.unique(np.linspace(0,len(vals)-1,min(1500,len(vals))).astype(int))
            a.plot(np.maximum(vals[ix],1e-12),(ix+1)/len(vals),label=label,color=color)
            cdfrows.extend(dict(dataset=d,metric=col,x=float(vals[i]),cdf=float((i+1)/len(vals))) for i in ix)
        a.set(xscale='log',xlim=(1e-6,1),ylim=(0,1),title=d,xlabel='Weak probability of actual teacher token',ylabel='Event CDF')
    ax.flat[0].legend(fontsize=8);fig.savefig(OUT/'probability_cdf.png');fig.savefig(OUT/'probability_cdf.pdf');plt.close(fig)
    pd.DataFrame(cdfrows).to_csv(OUT/'probability_cdf.csv',index=False)
    fig,ax=plt.subplots(2,4,figsize=(16,7),layout='constrained')
    for a,d in zip(ax.flat,ds):
        g=df[df.dataset==d]
        for col,label in [('w_rank_min','Best tie rank'),('w_rank_max','Worst tie rank')]:
            vals=np.sort(g[col].to_numpy());ix=np.unique(np.linspace(0,len(vals)-1,min(1500,len(vals))).astype(int))
            a.plot(vals[ix],(ix+1)/len(vals),label=label)
        a.set(xscale='log',ylim=(0,1),title=d,xlabel='Weak rank of actual teacher token',ylabel='Event CDF')
    ax.flat[0].legend(fontsize=8);fig.savefig(OUT/'rank_cdf.png');fig.savefig(OUT/'rank_cdf.pdf');plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for a,(label,g) in zip(ax,df.groupby('support_group',sort=True)):
        for col,lab in [('entropy_delta','Raw'),('masked_entropy_delta','Exclude original token')]:
            vals=np.sort(g[col].to_numpy());ix=np.unique(np.linspace(0,len(vals)-1,min(1500,len(vals))).astype(int))
            a.plot(vals[ix],(ix+1)/len(vals),label=lab)
        a.axvline(0,color='gray',ls='--');a.set(title=label,xlabel='Weak minus strong entropy (nats)',ylabel='Event CDF')
    ax[0].legend(fontsize=8);fig.savefig(OUT/'entropy_by_support.png');fig.savefig(OUT/'entropy_by_support.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    hb=axes[0].hexbin(np.log10(np.maximum(df.w_py,1e-15)),np.log10(df.w_rank_min),gridsize=65,bins='log',mincnt=1,cmap='viridis')
    fig.colorbar(hb,ax=axes[0],label='Events (log scale)')
    axes[0].set(xlabel='log10 weak probability of teacher token',ylabel='log10 weak rank (best tie position)')
    hb=axes[1].hexbin(df.w_entropy,df.s_entropy,gridsize=65,bins='log',mincnt=1,cmap='viridis')
    fig.colorbar(hb,ax=axes[1],label='Events (log scale)')
    lim=max(df.w_entropy.max(),df.s_entropy.max());axes[1].plot([0,lim],[0,lim],color='#bf4b45',ls='--')
    axes[1].set(xlabel='Weak entropy (nats)',ylabel='Strong entropy (nats)')
    fig.savefig(OUT/'rank_entropy_joint.png');fig.savefig(OUT/'rank_entropy_joint.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    tab=pd.DataFrame(summary).set_index('dataset').loc[ds];bottom=np.zeros(len(ds))
    for c,color in [('high','#208b83'),('middle','#8c949b'),('low','#cb614c')]:
        axes[0].barh(ds,tab[c],left=bottom,label=c,color=color);bottom+=tab[c].to_numpy()
    axes[0].set(xlabel='Fraction of intervention events',xlim=(0,1));axes[0].legend(loc='lower left',bbox_to_anchor=(0,1),ncol=3)
    for col,lab in [('entropy_delta','Raw entropy difference'),('masked_entropy_delta','After excluding original token')]:
        v=np.sort(df[col]);ix=np.unique(np.linspace(0,len(v)-1,1500).astype(int));axes[1].plot(v[ix],(ix+1)/len(v),label=lab)
    axes[1].axvline(0,color='gray',ls='--');axes[1].set(xlabel='Weak minus strong entropy (nats)',ylabel='Event CDF');axes[1].legend(fontsize=8)
    fig.savefig(OUT/'support_entropy_summary.png');fig.savefig(OUT/'support_entropy_summary.pdf');plt.close(fig)
    # Blind review pack: group labels and outcomes live only in the separate key.
    sampled=[]
    for (d,s),g in df.groupby(['dataset','support_group']):
        g=g.copy();g['sample_hash']=[hashlib.sha256(f'20260921:{i}'.encode()).hexdigest() for i in g.event_id]
        sampled.extend(g.sort_values('sample_hash').head(5).to_dict('records'))
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('/storage/yukaichengLab/share/model/Qwen3-1.7B',local_files_only=True)
    blind=[];key=[]
    for r in sampled:
        x=json.loads(Path(r['source']).read_text());assert sha(r['source'])==r['source_sha256']
        j=int(r['response_position']);prefix=x['prompt_token_ids']+x['response_token_ids'][:j]
        rid=hashlib.sha256(f'review:{r["event_id"]}'.encode()).hexdigest()[:12]
        blind.append(dict(review_id=rid,dataset=r['dataset'],problem=x.get('problem'),prefix_tail=tok.decode(prefix[-768:]),
             original_token=r['student_token_text'],replacement_token=r['teacher_token_text'],label=None,reviewer=None,
             instruction='Classify formatting/EOS/numeric-operation/reference/proposition-strategy/unclear; prefix and one-token choices only.'))
        key.append(dict(review_id=rid,event_id=int(r['event_id']),support_group=r['support_group'],weak_probability=r['w_py'],weak_rank=r['w_rank_min']))
    save(OUT/'semantic_review_blind.json',sorted(blind,key=lambda x:x['review_id']))
    save(OUT/'semantic_review_key.json',key)
    pooled=basic(df);macro=pd.DataFrame(intervals);macro=macro[macro.dataset=='MACRO']
    save(OUT/'headline.json',dict(pooled_event=pooled,dataset_macro=macro.to_dict('records'),semantic_review_cases=len(blind),
        semantic_review_status='Prepared, not human annotated; lexical categories are not logical-path labels'))
    save(OUT/'ANALYSIS_COMPLETE.json',dict(events=len(df),traces=len(tr),archive_verified=True,
          input_sha256=receipt['events_sha256'],summary_script_sha256=sha(__file__),bootstrap_repeats=2000,
          semantic_annotation_completed=False))
    print(json.dumps(pooled),flush=True)

if __name__=='__main__':main()
