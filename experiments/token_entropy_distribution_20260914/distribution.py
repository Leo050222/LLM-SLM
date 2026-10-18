"""Full response-token entropy for both models on identical frozen contexts."""
import argparse
import collections
import csv
import hashlib
import json
import math
import time
from pathlib import Path

SEED=20260914
DATASETS=['math500','olympiad_bench','bbh','cruxeval_o','cruxeval_i','humaneval_plus','mbpp_plus','livecodebench']
KS=[1,4,8,16,32]
TS=[2,2.5,3,3.5,4]

def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()

def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False));tmp.replace(p)

def prepare(a):
    tasks=[];selection={};manifests={}
    for kind,relative in [('math','opd_eval_hard/nojudge_grid_20260912'),('code','opd_eval_code/code_reasoning_grid_20260913')]:
        root=a.root/relative
        rows=[json.loads(l) for l in (root/'prompts.jsonl').read_text().splitlines() if l]
        chosen=[]
        for ds in DATASETS:
            pool=[r for r in rows if r['dataset']==ds]
            if not pool:continue
            sample=sorted(pool,key=lambda r:digest([SEED,ds,r['problem_id']]))[:a.n]
            assert len(sample)==a.n
            selection[ds]=[r['problem_id'] for r in sample];chosen+=sample
        for t in TS:
            for k in KS:
                config=f'k{k}_t{t:g}'
                group=root/(config.replace('.','p') if kind=='math' else config)
                m=json.loads((group/'manifest.json').read_text())
                assert m['thinking'] is False and m['first_token_ban'] is True
                assert m['sampling']==dict(temperature=1.,top_p=1.)
                assert m['settings']['teacher_tokens']==k and m['settings']['gap_threshold']==t
                manifests[str(group)]=digest(m)
                for r in chosen:
                    p=group/'completed'/f"{r['dataset_index']:05d}.json"
                    blob=p.read_bytes();x=json.loads(blob)
                    assert x['status']=='COMPLETE' and x['judge_calls']==0
                    assert all(x[key]==r[key] for key in ['dataset','problem_id','prompt_token_ids'])
                    assert len(x['response_token_ids'])==len(x['token_sources'])
                    assert set(x['token_sources']) <= {'student','teacher'}
                    tasks.append(dict(task_id=digest([config,r['dataset'],r['problem_id']])[:24],
                        dataset=r['dataset'],problem_id=r['problem_id'],config=config,k=k,threshold=t,
                        source=str(p),source_sha256=hashlib.sha256(blob).hexdigest(),
                        n_tokens=len(x['response_token_ids']),n_prompt=len(x['prompt_token_ids']),
                        finish_reason=x.get('finish_reason')))
    assert len(tasks)==8*a.n*25
    data=dict(seed=SEED,n_per_dataset=a.n,selection=selection,source_manifests=manifests,tasks=tasks,
              total_tokens=sum(t['n_tokens'] for t in tasks),protocol='full_response_entropy_v1')
    data['manifest_hash']=digest(data)
    p=a.out/'inputs.json'
    if p.exists():assert json.loads(p.read_text())==data
    else:save(p,data)
    print(json.dumps(dict(traces=len(tasks),tokens=data['total_tokens'],empty=sum(t['n_tokens']==0 for t in tasks))),flush=True)

def load_trace(task):
    blob=Path(task['source']).read_bytes()
    assert hashlib.sha256(blob).hexdigest()==task['source_sha256']
    return json.loads(blob)

def score(a):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    data=json.loads((a.out/'inputs.json').read_text())
    model_path='/storage/yukaichengLab/share/model/Qwen3-'+('1.7B' if a.role=='student' else '14B')
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    student_tok=AutoTokenizer.from_pretrained('/storage/yukaichengLab/share/model/Qwen3-1.7B',local_files_only=True)
    assert tok.get_vocab()==student_tok.get_vocab()
    torch.manual_seed(SEED)
    x=torch.randn(256,256,device='cuda');assert torch.isfinite(x@x).all();del x
    print('CUDA_WITNESS_OK',torch.cuda.get_device_name(),flush=True)
    model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,dtype=torch.bfloat16,
                                            attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    save(a.out/f'{a.role}_runtime.json',dict(model=model_path,vocab=digest(tok.get_vocab()),
         manifest_hash=data['manifest_hash'],torch=torch.__version__,vocab_size=model.config.vocab_size))
    started=time.monotonic();count=0;tokens=0;witness=False
    with torch.inference_mode():
        for i,task in enumerate(data['tasks']):
            target=a.out/a.role/(task['task_id']+'.npz')
            if target.exists():
                with np.load(target) as z:
                    assert str(z['manifest_hash'])==data['manifest_hash'] and len(z['entropy'])==task['n_tokens']
                continue
            row=load_trace(task);response=row['response_token_ids'];prompt=row['prompt_token_ids']
            n=len(response);values=[];t0=time.monotonic()
            if n:
                # Each value predicts observed response[j], conditioned on prompt+response[:j].
                ids=(prompt+response)[:-1]
                assert prompt and len(ids)<=40960
                input_ids=torch.tensor([ids],device='cuda',dtype=torch.long)
                hidden=model.model(input_ids=input_ids,use_cache=False,return_dict=True).last_hidden_state[0]
                start=len(prompt)-1
                for offset in range(0,n,64):
                    logits=model.lm_head(hidden[start+offset:start+min(n,offset+64)]).float()
                    lp=torch.log_softmax(logits,dim=-1)
                    h=-(lp.exp()*lp).sum(-1)
                    assert torch.isfinite(h).all() and (h>=0).all() and (h<=math.log(logits.shape[-1])+1e-4).all()
                    values.extend(h.tolist());del logits,lp,h
                del hidden
                if not witness:
                    checks=[]
                    for j in sorted({0,n//2,n-1}):
                        last=model.model(input_ids=input_ids[:,:len(prompt)+j],use_cache=False,
                                         return_dict=True).last_hidden_state[:,-1]
                        lp=torch.log_softmax(model.lm_head(last).float(),dim=-1)
                        h=-(lp.exp()*lp).sum().item();delta=abs(h-values[j])
                        assert delta<.05,('causal_index_alignment',j,delta)
                        checks.append(dict(response_position=j,delta_nats=delta))
                        del last,lp
                    save(a.out/f'{a.role}_alignment_witness.json',dict(passed=True,checks=checks))
                    witness=True
                del input_ids
            assert len(values)==n
            target.parent.mkdir(parents=True,exist_ok=True)
            tmp=target.with_suffix('.tmp')
            with tmp.open('wb') as f:
                np.savez_compressed(f,entropy=np.asarray(values,dtype=np.float32),
                    manifest_hash=data['manifest_hash'],seconds=time.monotonic()-t0)
            tmp.replace(target);count+=1;tokens+=n
            if count%20==0 or count==1:
                elapsed=time.monotonic()-started
                print(json.dumps(dict(role=a.role,completed=i+1,total=len(data['tasks']),tokens=tokens,
                    seconds=elapsed,tokens_per_second=tokens/elapsed)),flush=True)
    save(a.out/f'{a.role}_COMPLETE.json',dict(tasks=len(data['tasks']),manifest_hash=data['manifest_hash']))

def summarize(a):
    import numpy as np
    data=json.loads((a.out/'inputs.json').read_text());edges=np.linspace(0,12.8,641)
    accum={};traces=[];events=[]
    def add(key,s,t):
        if not len(s):return
        if key not in accum:
            accum[key]=dict(n=0,traces=0,s=0.,t=0.,higher=0,hs=np.zeros(640,dtype=np.int64),
                ht=np.zeros(640,dtype=np.int64),ws=np.zeros(640),wt=np.zeros(640),
                trace_s=0.,trace_t=0.,trace_higher=0.)
        q=accum[key];n=len(s);hs=np.histogram(s,edges)[0];ht=np.histogram(t,edges)[0]
        assert hs.sum()==ht.sum()==n
        q['n']+=n;q['traces']+=1;q['s']+=float(s.sum(dtype=np.float64));q['t']+=float(t.sum(dtype=np.float64))
        q['higher']+=int((s>t).sum());q['hs']+=hs;q['ht']+=ht;q['ws']+=hs/n;q['wt']+=ht/n
        q['trace_s']+=float(s.mean());q['trace_t']+=float(t.mean());q['trace_higher']+=float((s>t).mean())
    for task in data['tasks']:
        vals=[]
        for role in ['student','teacher']:
            with np.load(a.out/role/(task['task_id']+'.npz')) as z:
                assert str(z['manifest_hash'])==data['manifest_hash']
                vals.append(z['entropy'].copy())
        s,t=vals;assert len(s)==len(t)==task['n_tokens']
        row=load_trace(task);sources=np.array(row['token_sources'])
        for scope,mask in [('all',np.ones(len(s),dtype=bool)),('student_generated',sources=='student'),('teacher_generated',sources=='teacher')]:
            for ds,config in [(task['dataset'],task['config']),(task['dataset'],'ALL'),('ALL',task['config']),('ALL','ALL')]:
                add((ds,config,scope),s[mask],t[mask])
        if len(s):
            traces.append(dict(task_id=task['task_id'],dataset=task['dataset'],config=task['config'],
                n_tokens=len(s),student_mean=float(s.mean()),teacher_mean=float(t.mean()),
                student_higher_fraction=float((s>t).mean()),finish_reason=task['finish_reason']))
        for event in row['events']:
            if 'bridge_token_ids' not in event:continue
            before=event['anchor_response_position'];after=before+len(event['bridge_token_ids'])
            assert row['response_token_ids'][before:after]==event['bridge_token_ids']
            if after>=len(s):continue
            events.append(dict(task_id=task['task_id'],dataset=task['dataset'],config=task['config'],
                actual_k=len(event['bridge_token_ids']),before_position=before,after_position=after,
                student_before=float(s[before]),teacher_before=float(t[before]),
                student_after=float(s[after]),teacher_after=float(t[after])))
    summary=[];hist=[]
    for key,q in sorted(accum.items()):
        base=dict(zip(['dataset','config','token_source'],key))
        z=dict(base,n_tokens=q['n'],n_traces=q['traces'],student_mean=q['s']/q['n'],teacher_mean=q['t']/q['n'],
            student_higher_fraction=q['higher']/q['n'],student_trace_equal_mean=q['trace_s']/q['traces'],
            teacher_trace_equal_mean=q['trace_t']/q['traces'],student_higher_trace_equal=q['trace_higher']/q['traces'])
        for role,h in [('student',q['hs']),('teacher',q['ht'])]:
            c=h.cumsum()
            for p in [10,25,50,75,90,95,99]:
                idx=min(639,int(np.searchsorted(c,q['n']*p/100)))
                z[f'{role}_p{p}_approx']=float((edges[idx]+edges[idx+1])/2)
            for threshold in [1,2,4]:
                z[f'{role}_ge{threshold}_fraction']=int(h[int(threshold/.02):].sum())/q['n']
        summary.append(z)
        for j in range(640):
            hist.append(dict(base,left=float(edges[j]),right=float(edges[j+1]),
                student_count=int(q['hs'][j]),teacher_count=int(q['ht'][j]),
                student_trace_equal_mass=float(q['ws'][j]/q['traces']),
                teacher_trace_equal_mass=float(q['wt'][j]/q['traces'])))
    for name,rows in [('summary',summary),('histograms',hist),('trace_summary',traces),('intervention_points',events)]:
        with (a.out/(name+'.csv')).open('w',newline='') as f:
            if rows:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(a.out/'COMPLETE.json',dict(tasks=len(data['tasks']),tokens=data['total_tokens'],
         manifest_hash=data['manifest_hash'],histogram_width_nats=.02,quantile_error_bound_nats=.01))
    print('COMPLETE',data['total_tokens'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','score','summarize'])
    p.add_argument('--root',type=Path,default=Path('/storage/yukaichengLab/wangxin/test_eiloeo'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--n',type=int,default=100)
    p.add_argument('--role',choices=['student','teacher']);a=p.parse_args();globals()[a.action](a)
