"""Frozen held-out teacher8 -> four prefix lengths -> student8; atomic bundles."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT=Path('/storage/yukaichengLab/wangxin/test_eiloeo')
SOURCE=ROOT/'opd_eval_deepmath/ge6_test1000_native_nonthinking_seed20260811/qwen3_1_7b/merged/rollouts.jsonl'
DATA=ROOT/'opd_data/deepmath_ge6_opd_train15pct_seed20260811/test_1000.parquet'
STUDENT='/storage/yukaichengLab/share/model/Qwen3-1.7B'
TEACHER='/storage/yukaichengLab/share/model/Qwen3-14B'
H=(32,64,128,256)
SEED=20260811

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False))
    tmp.replace(p)
def load(p):return json.loads(p.read_text())
def source(limit):
    assert sha(DATA)=='6ef8e66d181b7522bad2904ad1fcf20d8fc152af0c9d7eb05e07a9f2f0415ae0'
    rows=sorted([json.loads(x) for x in SOURCE.read_text().splitlines() if x.strip()],key=lambda r:int(r['dataset_index']))
    assert len(rows)==1000 and [int(r['dataset_index']) for r in rows]==list(range(1000))
    assert len({str(r['problem_id']) for r in rows})==1000
    return rows[:limit]
def teacher_path(out,i):return out/'teacher'/f'{i:04d}.json'
def student_path(out,h,i,t):return out/f'student/h{h}'/f'{i:04d}_{t}.json'
def validate_teacher(bundle,r):
    assert bundle['dataset_index']==r['dataset_index'] and bundle['problem_id']==r['problem_id']
    assert bundle['prompt_token_ids']==r['prompt_token_ids'] and bundle['solution']==r['solution']
    assert [x['teacher_replicate'] for x in bundle['outputs']]==list(range(8))
    assert all(isinstance(x['token_ids'],list) and x['token_ids'] for x in bundle['outputs'])
def validate_student(bundle,r,t,h):
    assert bundle['dataset_index']==r['dataset_index'] and bundle['problem_id']==r['problem_id']
    assert bundle['solution']==r['solution'] and bundle['teacher_replicate']==t['teacher_replicate']
    assert bundle['h']==h and bundle['prefix_token_ids']==t['token_ids'][:h]
    assert [x['student_replicate'] for x in bundle['outputs']]==list(range(8))

def prepare(out,limit):
    rows=source(limit)
    cfg={'source_sha256':sha(SOURCE),'test_sha256':sha(DATA),'script_sha256':sha(__file__),'scorer_sha256':sha(ROOT/'opd_scripts/run_opd_logit_diagnostic.py'),'teacher':TEACHER,'student':STUDENT,'tokenizer':STUDENT,'limit':limit,'lengths':list(H),'teacher_n':8,'student_n':8,'seed':SEED,'temperature':1,'top_p':1,'top_k':-1,'thinking':False,'total_response_cap':40264,'max_model_len':40960,'short_teacher_policy':'if naturally finished by cutoff, retain teacher answer in 8 bookkeeping rows and label no student generation'}
    cp=out/'config.json'
    if cp.exists():assert load(cp)==cfg, 'Configuration mismatch'
    else:atomic(cp,cfg)
    print('PREFLIGHT_OK',limit,flush=True)
    return rows

def verify(out,rows,stage):
    digest=hashlib.sha256()
    count=0
    for r in rows:
        i=int(r['dataset_index']);tp=teacher_path(out,i);t=load(tp)
        validate_teacher(t,r);digest.update(bytes.fromhex(sha(tp)))
        if stage=='teacher':count+=8
        else:
            for h in H:
                for teacher in t['outputs']:
                    p=student_path(out,h,i,teacher['teacher_replicate']);b=load(p)
                    validate_student(b,r,teacher,h)
                    digest.update(bytes.fromhex(sha(p)));count+=8
    expected=len(rows)*8*(1 if stage=='teacher' else 32)
    assert count==expected
    atomic(out/f'{stage.upper()}_GENERATION_COMPLETE.json',{'rows':count,'bundle_digest':digest.hexdigest(),'questions':len(rows)})
    print('STRICT_GENERATION_COMPLETE',stage,count,flush=True)

def generate(args,rows):
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from vllm.inputs import TokensPrompt
    from run_opd_logit_diagnostic import deepmath_user_prompt
    import torch
    tok=AutoTokenizer.from_pretrained(STUDENT,local_files_only=True)
    torch.manual_seed(SEED)
    assert torch.cuda.device_count()==1 and int(torch.arange(8,device='cuda').sum())==28
    print('CUDA_WITNESS_OK',torch.cuda.get_device_name(),flush=True)
    selected=[r for r in rows if int(r['dataset_index'])%args.shards==args.rank]
    tasks=[]
    for r in selected:
        encoded=tok.apply_chat_template([{'role':'user','content':deepmath_user_prompt(r['problem'])}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
        ids=encoded['input_ids'] if hasattr(encoded,'keys') else encoded
        assert list(ids)==r['prompt_token_ids']
        i=int(r['dataset_index'])
        if args.mode=='teacher':
            p=teacher_path(args.output,i)
            if p.exists():validate_teacher(load(p),r);continue
            tasks.append((r,None,None,p))
        else:
            b=load(teacher_path(args.output,i));validate_teacher(b,r)
            for h in H:
                for t in b['outputs']:
                    p=student_path(args.output,h,i,t['teacher_replicate'])
                    if p.exists():validate_student(load(p),r,t,h);continue
                    prefix=t['token_ids'][:h]
                    finished=(len(t['token_ids'])<=h and t['finish_reason']=='stop') or tok.eos_token_id in prefix
                    if finished:
                        item={'dataset_index':i,'problem_id':r['problem_id'],'solution':r['solution'],'h':h,'teacher_replicate':t['teacher_replicate'],'prefix_token_ids':prefix,'teacher_finished':True,'outputs':[{'student_replicate':s,'token_ids':[],'response':t['response'],'finish_reason':'teacher_stop'} for s in range(8)]}
                        atomic(p,item)
                    else:tasks.append((r,t,h,p))
    if not tasks:print('ALL_BUNDLES_REUSED',args.mode,args.rank,flush=True);return
    engine=LLM(model=TEACHER if args.mode=='teacher' else STUDENT,tokenizer=STUDENT,dtype='bfloat16',tensor_parallel_size=1,seed=SEED,max_model_len=40960,max_num_seqs=8,gpu_memory_utilization=.9,enable_prefix_caching=True)
    for start in range(0,len(tasks),8):
        batch=tasks[start:start+8];prompts=[];params=[]
        for r,t,h,p in batch:
            prefix=[] if t is None else t['token_ids'][:h]
            ids=r['prompt_token_ids']+prefix
            cap=min(40264-len(prefix),40960-len(ids));assert cap>0
            prompts.append(TokensPrompt(prompt_token_ids=ids))
            params.append(SamplingParams(temperature=1,top_p=1,top_k=-1,n=8,seed=SEED,max_tokens=cap))
        outputs=engine.generate(prompts,params,use_tqdm=False)
        for (r,t,h,p),output in zip(batch,outputs,strict=True):
            assert len(output.outputs)==8
            b={'dataset_index':int(r['dataset_index']),'problem_id':r['problem_id'],'solution':r['solution'],'outputs':[]}
            prefix=[] if t is None else t['token_ids'][:h]
            if t is None:b['prompt_token_ids']=r['prompt_token_ids']
            else:b.update(h=h,teacher_replicate=t['teacher_replicate'],prefix_token_ids=prefix,teacher_finished=False)
            for j,g in enumerate(output.outputs):
                ids=list(g.token_ids)
                b['outputs'].append({('teacher_replicate' if t is None else 'student_replicate'):j,'token_ids':ids,'response':tok.decode(prefix+ids,skip_special_tokens=False),'finish_reason':g.finish_reason})
            if t is None:validate_teacher(b,r)
            else:validate_student(b,r,t,h)
            atomic(p,b)
        print('PROGRESS',args.mode,args.rank,min(start+8,len(tasks)),len(tasks),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','teacher','student','verify-teacher','verify-student']);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--limit',type=int,default=1000);ap.add_argument('--rank',type=int,default=0);ap.add_argument('--shards',type=int,default=8)
    args=ap.parse_args();assert 0<args.limit<=1000 and 0<=args.rank<args.shards
    if args.mode=='prepare':prepare(args.output,args.limit);return
    rows=source(args.limit)
    cfg=load(args.output/'config.json');assert cfg['script_sha256']==sha(__file__) and cfg['source_sha256']==sha(SOURCE) and cfg['limit']==args.limit
    if args.mode.startswith('verify-'):verify(args.output,rows,args.mode.split('-')[1])
    else:generate(args,rows)
if __name__=='__main__':main()
