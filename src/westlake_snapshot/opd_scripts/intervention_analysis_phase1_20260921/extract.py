import argparse,csv,hashlib,json,math,os,time,unicodedata
from pathlib import Path

ROOT=Path('/storage/yukaichengLab/wangxin/test_eiloeo')
ARCHIVE=ROOT/'opd_analysis/intervention_logits_bf16_x8_20260920'
OUT=ROOT/'opd_analysis/intervention_analysis_phase1_20260921'
SPEC=dict(version=1,archive_manifest='ced4ae7505ffea1e401b3e262b68c5bec38574ede3217d658396817033b4f75a',
          temperature=1,softmax_dtype='float32',high_support='w_rank_max<=5 and w_py>=0.01',
          low_support='w_rank_min>50 and w_py<0.001',bootstrap_seed=20260921,bootstrap_repeats=2000,
          tie_rule='strictly greater / greater-or-equal rank bounds; nucleus all possible tie orders',
          weights='event; equal question then equal nonempty replicate then equal event; dataset macro')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)

def metrics(z,x,y,topk=64):
    import torch
    z=z.float();n,v=z.shape;ix=torch.arange(n,device=z.device)
    assert torch.isfinite(z).all()
    lp=z.log_softmax(-1);p=lp.exp();zy=z[ix,y,None]
    greater=z>zy;equal=z==zy
    rmin=1+greater.sum(-1);rmax=(greater|equal).sum(-1)
    mass=(p*greater).sum(-1);tie=(p*equal).sum(-1)
    entropy=-(p*lp).sum(-1)
    assert ((p.sum(-1)-1).abs()<1e-5).all()
    assert ((entropy>=0)&(entropy<=math.log(v)+1e-5)).all()
    out=dict(logpx=lp[ix,x],logpy=lp[ix,y],px=p[ix,x],py=p[ix,y],
             entropy=entropy,rank_min=rmin,rank_max=rmax,tie_count=equal.sum(-1),
             mass_above=mass,tie_mass=tie,log_y_over_x=lp[ix,y]-lp[ix,x],
             log_y_over_top1=lp[ix,y]-lp.max(-1).values)
    for k in [1,5,10,50]:
        out[f'top{k}_certain']=rmax<=k;out[f'top{k}_possible']=rmin<=k
    for a in [.9,.95]:
        out[f'nucleus{int(a*100)}_certain']=mass+tie-p[ix,y]<a
        out[f'nucleus{int(a*100)}_possible']=mass<a
    tv,ti=torch.topk(z,min(topk,v),dim=-1)
    arrays=dict(top_ids=ti,top_logprobs=lp.gather(1,ti))
    # Historical top16 comparisons must be made before masking.
    original_lp=lp
    masked=z.clone();masked[ix,x]=-torch.inf
    qlp=masked.log_softmax(-1);q=qlp.exp()
    out.update(qy=q[ix,y],logqy=qlp[ix,y],masked_entropy=-(q*torch.where(torch.isfinite(qlp),qlp,0.)).sum(-1),
               masked_rank_min=rmin-(z[ix,x]>z[ix,y]).long(),masked_rank_max=rmax-(z[ix,x]>=z[ix,y]).long())
    _,qi=torch.topk(masked,min(topk,v),dim=-1)
    arrays.update(masked_top_ids=qi,masked_top_logprobs=qlp.gather(1,qi))
    assert (out['rank_min']<=out['rank_max']).all() and (x!=y).all()
    assert torch.isfinite(out['masked_entropy']).all()
    return out,arrays,original_lp,ti

def selftest(device):
    import torch
    z=torch.tensor([[0.,0.,-2.,-3.],[100.,0.,-100.,-1000.]],device=device)
    x=torch.tensor([0,0],device=device);y=torch.tensor([1,1],device=device)
    m,a,_,_=metrics(z,x,y)
    assert m['rank_min'].tolist()==[1,2] and m['rank_max'].tolist()==[2,2]
    assert not m['top1_certain'][0] and m['top1_possible'][0]
    assert abs(m['qy'][1].item()-1)<1e-6
    for i in range(2):
        p=z[i].double().softmax(-1);h=-(p*p.clamp_min(1e-300).log()).sum()
        assert abs(h.item()-m['entropy'][i].item())<1e-5
        ids=[j for j in range(4) if j!=x[i].item()]
        q=z[i,ids].double().softmax(-1)
        hq=-(q*q.clamp_min(1e-300).log()).sum()
        assert abs(hq.item()-m['masked_entropy'][i].item())<1e-5
    print('METRIC_SELFTEST_PASSED',device,flush=True)

def token_class(token,text,eos):
    if token==eos:return 'EOS'
    if not text or text.isspace():return 'whitespace'
    s=text.strip()
    if all(unicodedata.category(c)[0] in 'PS' for c in s):return 'punctuation_symbol'
    if any(c.isdigit() for c in s):return 'contains_digit'
    if s.isidentifier():return 'word_or_identifier'
    return 'mixed_text'

def score_info(row):
    fn=Path(row['source']).name
    if row['family']=='code':
        path=ROOT/f'opd_eval_code/code_grid_x8_20260916/r{row["replicate"]}_k1_t2/scores'/fn
    elif row['replicate']==0:
        path=ROOT/'opd_rescore/grid_math_gpt4o_mini_openlux_20260916/scores'/row['dataset']/'k1_t2'/fn
    else:path=Path(row['source']).parent.parent/'scores'/fn
    if not path.exists():return dict(correct=None,score_status='missing',score_source=str(path),score_signature=None)
    s=json.loads(path.read_text())
    assert s['source_sha256']==row['source_sha256'] and s['dataset']==row['dataset'] and s['problem_id']==row['problem_id']
    c=s.get('correct',s.get('is_correct'))
    assert type(c) is bool
    if row['finish_reason']=='trace_timeout':c=False
    status=s.get('status','unspecified')
    if not isinstance(status,str):status=json.dumps(status,sort_keys=True,ensure_ascii=False)
    return dict(correct=c,score_status=status,score_source=str(path),
                score_signature=json.dumps(s.get('scoring_signature',s.get('judge_signature')),sort_keys=True,ensure_ascii=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--test',action='store_true');p.add_argument('--device',default='cuda');a=p.parse_args()
    # Only one GPU is allocated. Under cgroups nvidia-smi may enumerate it as zero.
    if a.device=='cuda' and os.environ.get('SLURMD_NODENAME')=='gnho008':
        import subprocess
        inv=list(csv.DictReader(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv'],text=True).splitlines(),skipinitialspace=True))
        assert len(inv)==1
        assert inv[0]['uuid']!='GPU-c33c45c6-61d6-36bc-e5ac-6d80e3bc4968'
        os.environ['CUDA_VISIBLE_DEVICES']=inv[0]['uuid']
    import torch,numpy as np,pyarrow as pa,pyarrow.parquet as pq
    from safetensors import safe_open
    torch.set_num_threads(8);selftest(a.device)
    if a.test:return
    from transformers import AutoTokenizer
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'extract.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    spec={**SPEC,'script_sha256':sha(__file__)}
    if (OUT/'metric_spec.json').exists():assert json.loads((OUT/'metric_spec.json').read_text())==spec
    else:save(OUT/'metric_spec.json',spec)
    m=json.loads((ARCHIVE/'manifest.json').read_text());verified=json.loads((ARCHIVE/'ALL_LOGITS_VERIFIED.json').read_text())
    assert sha(ARCHIVE/'manifest.json')==SPEC['archive_manifest']==verified['manifest_sha256']
    tok=AutoTokenizer.from_pretrained('/storage/yukaichengLab/share/model/Qwen3-1.7B',local_files_only=True)
    cache={};scalars=[];start=time.monotonic();trace_inventory=[];done=0
    for si,shard in enumerate(m['shards']):
        dest=OUT/'shards'/(Path(shard['name']).stem+'.parquet');receipt=dest.with_suffix('.json')
        dest.parent.mkdir(exist_ok=True)
        inp=ARCHIVE/'inputs'/shard['name'];assert sha(inp)==shard['sha256']
        rows=[json.loads(l) for l in inp.read_text().splitlines()]
        events=[]
        for row in rows:
            info=score_info(row)
            base={k:row[k] for k in ['dataset','problem_id','replicate','source','source_sha256','finish_reason','response_length','prompt_length']}
            base.update(info)
            trace_inventory.append({**base,'interventions':len(row['events'])})
            for ordinal,e in enumerate(row['events']):
                events.append({**base,**e,'intervention_ordinal':ordinal,'relative_position':e['response_position']/max(1,row['response_length'])})
        ids=[e['event_id'] for e in events]
        assert ids==list(range(done,done+len(events)))
        if receipt.exists():
            rec=json.loads(receipt.read_text());assert rec['sha256']==sha(dest) and rec['script_sha256']==spec['script_sha256']
        else:
            data={k:[e[k] for e in events] for k in events[0] if k not in ['historical_teacher_top_ids','historical_teacher_top_logprobs']}
            for key in ['student_token_id','teacher_token_id']:
                for t in data[key]:
                    if t not in cache:cache[t]=tok.decode([t])
                data[key.replace('_id','_text')]=[cache[t] for t in data[key]]
            data['token_class']=[token_class(e['teacher_token_id'],cache[e['teacher_token_id']],tok.eos_token_id) for e in events]
            for role,tag in [('student','w'),('teacher','s')]:
                path=ARCHIVE/role/(Path(shard['name']).stem+'.safetensors')
                with safe_open(str(path),framework='pt',device='cpu') as f:
                    assert f.get_tensor('event_ids').tolist()==ids
                    assert f.metadata()['manifest_sha256']==SPEC['archive_manifest']
                    assert f.get_slice('logits').get_dtype()=='BF16'
                    for offset in range(0,len(events),64):
                        es=events[offset:offset+64];z=f.get_slice('logits')[offset:offset+len(es)].to(a.device)
                        x=torch.tensor([e['student_token_id'] for e in es],device=a.device)
                        y=torch.tensor([e['teacher_token_id'] for e in es],device=a.device)
                        vals,arrays,lp,top=metrics(z,x,y)
                        if role=='teacher':
                            oldid=torch.tensor([e['historical_teacher_top_ids'] for e in es],device=a.device)
                            oldlp=torch.tensor([e['historical_teacher_top_logprobs'] for e in es],device=a.device)
                            d=lp.gather(1,oldid)-oldlp
                            vals.update(historical_top16_abs_error_mean=d.abs().mean(-1),historical_top16_abs_error_max=d.abs().max(-1).values,
                                        historical_top16_overlap=(top[:,:16,None]==oldid[:,None,:]).any(-1).float().mean(-1))
                            histz=z.gather(1,oldid).float();cut=z.gather(1,top[:,15:16]).float()
                            vals['historical_top16_at_or_above_cutoff']=(histz>=cut).float().mean(-1)
                        for k,v in {**vals,**arrays}.items():data.setdefault(tag+'_'+k,[]).extend(v.cpu().tolist())
                        del z,vals,arrays,lp,top
            df={**data}
            df['support_group']=['high' if r<=5 and p>=.01 else 'low' if l>50 and p<.001 else 'middle'
                    for r,l,p in zip(data['w_rank_max'],data['w_rank_min'],data['w_py'])]
            df['entropy_delta']=(np.array(data['w_entropy'])-np.array(data['s_entropy'])).tolist()
            df['masked_entropy_delta']=(np.array(data['w_masked_entropy'])-np.array(data['s_masked_entropy'])).tolist()
            for tag,role in [('w','student'),('s','teacher')]:
                df[tag+'_replay_logpx_error']=(np.array(data[tag+'_logpx'])-np.array(data['historical_'+role+'_logprob'])).tolist()
            df['replay_gap']=(np.array(data['w_logpx'])-np.array(data['s_logpx'])).tolist()
            tmp=dest.with_suffix('.tmp');pq.write_table(pa.Table.from_pydict(df),tmp,compression='zstd');tmp.replace(dest)
            save(receipt,dict(sha256=sha(dest),events=len(events),script_sha256=spec['script_sha256']))
        table=pq.read_table(dest)
        cols=[c for c in table.column_names if not any(c.endswith(s) for s in ['top_ids','top_logprobs'])]
        scalars.append(table.select(cols));done+=len(events)
        save(OUT/'PROGRESS.json',dict(events=done,total=399601,shards=si+1,seconds=time.monotonic()-start,updated=time.time()))
        print('SHARD',si+1,'EVENTS',done,'SECONDS',round(time.monotonic()-start,1),flush=True)
    assert done==399601
    pq.write_table(pa.concat_tables(scalars,promote_options='default'),OUT/'events.parquet',compression='zstd')
    for row in json.loads((ARCHIVE/'no_intervention_traces.json').read_text()):
        trace_inventory.append({**{k:v for k,v in row.items() if k in ['dataset','problem_id','replicate','source','source_sha256','finish_reason','response_length','prompt_length']},**score_info(row),'interventions':0})
    assert len(trace_inventory)==36568
    pq.write_table(pa.Table.from_pylist(trace_inventory),OUT/'traces.parquet',compression='zstd')
    save(OUT/'EXTRACTION_COMPLETE.json',dict(events=done,traces=len(trace_inventory),seconds=time.monotonic()-start,
              events_sha256=sha(OUT/'events.parquet'),traces_sha256=sha(OUT/'traces.parquet'),spec=spec))

if __name__=='__main__':main()
