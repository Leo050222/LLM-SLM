"""Paired raw-distribution Top-1 agreement after retained one-token help."""
import argparse,csv,hashlib,json,time,collections
from pathlib import Path

def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False));tmp.replace(p)

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()

def prepare(a):
    original=json.loads((a.root/'opd_analysis/token_entropy_distribution_20260914/inputs.json').read_text())
    tasks=[];empty=0
    for t in original['tasks']:
        if t['k']!=1:continue
        blob=Path(t['source']).read_bytes();assert hashlib.sha256(blob).hexdigest()==t['source_sha256']
        r=json.loads(blob);n=len(r['response_token_ids']);points=[]
        for e in r['events']:
            if 'bridge_token_ids' not in e:continue
            assert len(e['bridge_token_ids'])==1
            pos=e['anchor_response_position'];assert r['response_token_ids'][pos:pos+1]==e['bridge_token_ids']
            if pos+1<n:points.append(pos+1)
        if points:tasks.append(dict(t,positions=points))
        else:empty+=1
    data=dict(tasks=tasks,excluded_traces=empty,source_manifest_hash=original['manifest_hash'],protocol='raw_top1_after_k1_v1')
    data['manifest_hash']=digest(data)
    p=a.out/'inputs.json'
    if p.exists():assert json.loads(p.read_text())==data
    else:save(p,data)
    print('PREPARED',len(tasks),sum(len(t['positions']) for t in tasks),flush=True)

def score(a):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    data=json.loads((a.out/'inputs.json').read_text());base='/storage/yukaichengLab/share/model/'
    name='Qwen3-1.7B' if a.role=='student' else 'Qwen3-14B'
    tok=AutoTokenizer.from_pretrained(base+name,local_files_only=True)
    st=AutoTokenizer.from_pretrained(base+'Qwen3-1.7B',local_files_only=True)
    assert tok.get_vocab()==st.get_vocab()
    torch.manual_seed(20260914);x=torch.randn(256,256,device='cuda');assert torch.isfinite(x@x).all();del x
    print('CUDA_WITNESS_OK',torch.cuda.get_device_name(),flush=True)
    model=AutoModelForCausalLM.from_pretrained(base+name,local_files_only=True,dtype=torch.bfloat16,
                                            attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False);start=time.monotonic();witness=False
    with torch.inference_mode():
        for i,t in enumerate(data['tasks']):
            target=a.out/a.role/(t['task_id']+'.json')
            if target.exists():
                assert json.loads(target.read_text())['manifest_hash']==data['manifest_hash'];continue
            blob=Path(t['source']).read_bytes();assert hashlib.sha256(blob).hexdigest()==t['source_sha256']
            r=json.loads(blob);prompt=r['prompt_token_ids'];offsets=t['positions']
            positions=[len(prompt)+p-1 for p in offsets]
            ids=torch.tensor([(prompt+r['response_token_ids'])[:max(positions)+1]],device='cuda',dtype=torch.long)
            hidden=model.model(input_ids=ids,use_cache=False,return_dict=True).last_hidden_state[0]
            selected=hidden[positions];del hidden
            points=[]
            for j in range(0,len(positions),64):
                logits=model.lm_head(selected[j:j+64]).float();lp=torch.log_softmax(logits,dim=-1)
                top_lp,top_id=lp.topk(2,dim=-1);entropy=-(lp.exp()*lp).sum(-1)
                assert torch.isfinite(entropy).all()
                for z in range(len(top_id)):
                    tid=int(top_id[z,0]);points.append(dict(response_position=offsets[j+z],token_id=tid,
                       token_text=tok.decode([tid]),pmax=float(top_lp[z,0].exp()),
                       runner_up_probability=float(top_lp[z,1].exp()),entropy=float(entropy[z])))
                del logits,lp,top_lp,top_id,entropy
            if not witness:
                h=model.model(input_ids=ids[:,:positions[0]+1],use_cache=False,return_dict=True).last_hidden_state[:,-1]
                lp=torch.log_softmax(model.lm_head(h).float(),dim=-1)
                delta=abs(float(lp.max().exp())-points[0]['pmax'])
                assert delta<.02
                save(a.out/f'{a.role}_witness.json',dict(passed=True,pmax_delta=delta,
                     same_top1=int(lp.argmax())==points[0]['token_id'],vocab_hash=digest(tok.get_vocab())))
                del h,lp;witness=True
            save(target,dict(manifest_hash=data['manifest_hash'],points=points));del ids,selected
            if i%50==0:print(json.dumps(dict(role=a.role,completed=i+1,total=len(data['tasks']),seconds=time.monotonic()-start)),flush=True)
    save(a.out/f'{a.role}_COMPLETE.json',dict(manifest_hash=data['manifest_hash'],tasks=len(data['tasks'])))

def summarize(a):
    data=json.loads((a.out/'inputs.json').read_text());rows=[];groups=collections.defaultdict(list)
    for t in data['tasks']:
        s,q=[json.loads((a.out/role/(t['task_id']+'.json')).read_text()) for role in ['student','teacher']]
        assert s['manifest_hash']==q['manifest_hash']==data['manifest_hash']
        assert len(s['points'])==len(t['positions'])
        for i,(sp,tp) in enumerate(zip(s['points'],q['points'],strict=True)):
            assert sp['response_position']==tp['response_position']==t['positions'][i]
            row=dict(task_id=t['task_id'],problem_id=t['problem_id'],dataset=t['dataset'],threshold=t['threshold'],
                first=i==0,response_position=sp['response_position'],agree=sp['token_id']==tp['token_id'],
                student_token_id=sp['token_id'],teacher_token_id=tp['token_id'],student_token=sp['token_text'],
                teacher_token=tp['token_text'],student_pmax=sp['pmax'],teacher_pmax=tp['pmax'],
                student_margin=sp['pmax']-sp['runner_up_probability'],teacher_margin=tp['pmax']-tp['runner_up_probability'],
                student_entropy=sp['entropy'],teacher_entropy=tp['entropy'])
            rows.append(row)
            for ds,threshold in [(t['dataset'],t['threshold']),(t['dataset'],'ALL'),('ALL',t['threshold']),('ALL','ALL')]:
                groups[ds,str(threshold),'all'].append(row)
                if i==0:groups[ds,str(threshold),'first'].append(row)
    summary=[]
    for key,rr in sorted(groups.items()):
        by=collections.defaultdict(list)
        for r in rr:by[r['task_id']].append(r)
        summary.append(dict(zip(['dataset','threshold','selection'],key),n=len(rr),n_traces=len(by),
            agreement=sum(r['agree'] for r in rr)/len(rr),
            trace_equal_agreement=sum(sum(r['agree'] for r in rs)/len(rs) for rs in by.values())/len(by),
            student_pmax=sum(r['student_pmax'] for r in rr)/len(rr),teacher_pmax=sum(r['teacher_pmax'] for r in rr)/len(rr)))
    for name,data_rows in [('paired_top1',rows),('summary',summary)]:
        with (a.out/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data_rows[0]));w.writeheader();w.writerows(data_rows)
    save(a.out/'COMPLETE.json',dict(tasks=len(data['tasks']),positions=len(rows),manifest_hash=data['manifest_hash']))
    print('COMPLETE',len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','score','summarize'])
    p.add_argument('--root',type=Path,default=Path('/storage/yukaichengLab/wangxin/test_eiloeo'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--role',choices=['student','teacher'])
    a=p.parse_args();globals()[a.action](a)
