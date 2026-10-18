import argparse,json,hashlib,os,multiprocessing as mp
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path('/storage/yukaichengLab/wangxin/test_eiloeo')
OLD=ROOT/'opd_eval_deepmath/ge6_test1000_student_only_nonthinking_x8_seed20260811/student_only'
MODEL='/storage/yukaichengLab/share/model/Qwen3-1.7B'
DATA=ROOT/'opd_data/deepmath_ge6_opd_train15pct_seed20260811/test_1000.parquet'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def atomic(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False));tmp.replace(p)
def config(out):
 c=load(out/'config.json');assert c['code_sha']==sha(Path(__file__));return c
def prepare(out):
 assert sha(DATA)=='6ef8e66d181b7522bad2904ad1fcf20d8fc152af0c9d7eb05e07a9f2f0415ae0'
 src=OLD/'scored/rollouts_scored.jsonl'
 rows=[json.loads(x) for x in src.read_text().splitlines() if x.strip()]
 assert len(rows)==8000 and sum(bool(r['correct']) for r in rows)==2247
 for rank in range(4):
  c=load(OLD/f'shards/shard_{rank}/run_config.json')
  assert c['student']==MODEL and c['tokenizer']==MODEL and c['thinking_mode']==False
  assert c['sampling']=={'temperature':1.0,'top_p':1.0,'top_k':-1,'max_tokens':40264,'max_model_len':40960,'n':8,'seed':20260811}
  assert c['output_sha256']==sha(OLD/f'shards/shard_{rank}/rollouts.jsonl')
 by={}
 for r in rows:
  i=int(r['dataset_index']);j=int(r['replicate_index']);assert (i,j) not in by;by[i,j]=r
 assert set(by)=={(i,j) for i in range(1000) for j in range(8)}
 cfg={'code_sha':sha(Path(__file__)),'old_scored_sha':sha(src),'scorer_sha':sha(ROOT/'opd_scripts/run_opd_logit_diagnostic.py'),'model':MODEL,'temperature':1,'top_p':1,'top_k':-1,'max_tokens':40264,'max_model_len':40960,'old_replicates':[0,7],'new_replicates':[8,63],'new_group_seeds':[20260910+g for g in range(7)],'thinking':False}
 if (out/'config.json').exists():assert load(out/'config.json')==cfg
 else:atomic(out/'config.json',cfg)
 for i in range(1000):
  rs=[by[i,j] for j in range(8)];first=rs[0]
  assert all(r['problem_id']==first['problem_id'] and r['prompt_token_ids']==first['prompt_token_ids'] and r['solution']==first['solution'] for r in rs)
  atomic(out/'old8'/f'{i:04d}.json',rs)
 print('PREPARED_OLD8',len(rows),flush=True)
def bundle(out,i,g):return out/'new56'/f'{i:04d}_{g}.json'
def validate(b,i,g,old):
 assert b['dataset_index']==i and b['problem_id']==old['problem_id'] and b['solution']==old['solution']
 assert b['group']==g and b['seed']==20260910+g
 assert [r['replicate_index'] for r in b['outputs']]==list(range(8+g*8,16+g*8))
def generate(out,rank,shards):
 config(out)
 from transformers import AutoTokenizer
 from vllm import LLM,SamplingParams
 from vllm.inputs import TokensPrompt
 from run_opd_logit_diagnostic import deepmath_user_prompt
 import torch
 assert torch.cuda.device_count()==1
 torch.manual_seed(20260910);assert int(torch.arange(8,device='cuda').sum())==28
 print('CUDA_OK',rank,torch.cuda.get_device_name(),flush=True)
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tasks=[]
 for i in range(1000):
  selected=[g for g in range(7) if (i*7+g)%shards==rank]
  if not selected:continue
  r=load(out/'old8'/f'{i:04d}.json')[0]
  enc=tok.apply_chat_template([{'role':'user','content':deepmath_user_prompt(r['problem'])}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
  assert list(enc['input_ids'] if hasattr(enc,'keys') else enc)==r['prompt_token_ids']
  for g in selected:
   p=bundle(out,i,g)
   if p.exists():validate(load(p),i,g,r)
   else:tasks.append((i,g,r,p))
 if not tasks:return
 engine=LLM(model=MODEL,tokenizer=MODEL,dtype='bfloat16',tensor_parallel_size=1,seed=20260910,max_model_len=40960,max_num_seqs=8,gpu_memory_utilization=.9,enable_prefix_caching=True)
 for start in range(0,len(tasks),8):
  batch=tasks[start:start+8]
  prompts=[TokensPrompt(prompt_token_ids=r['prompt_token_ids']) for i,g,r,p in batch]
  params=[SamplingParams(n=8,seed=20260910+g,temperature=1,top_p=1,top_k=-1,max_tokens=40264) for i,g,r,p in batch]
  outputs=engine.generate(prompts,params,use_tqdm=False)
  for (i,g,r,p),o in zip(batch,outputs,strict=True):
   assert len(o.outputs)==8
   b={'dataset_index':i,'problem_id':r['problem_id'],'solution':r['solution'],'group':g,'seed':20260910+g,'outputs':[{'replicate_index':8+g*8+j,'response_token_ids':list(v.token_ids),'response':tok.decode(v.token_ids,skip_special_tokens=False),'finish_reason':v.finish_reason} for j,v in enumerate(o.outputs)]}
   validate(b,i,g,r);atomic(p,b)
  print('GEN_PROGRESS',rank,min(start+8,len(tasks)),len(tasks),flush=True)
def verify(out):
 config(out)
 for i in range(1000):
  old=load(out/'old8'/f'{i:04d}.json')[0]
  for g in range(7):validate(load(bundle(out,i,g)),i,g,old)
 atomic(out/'GENERATION_COMPLETE.json',{'old':8000,'new':56000,'total':64000})
def score_one(task):
 from run_opd_logit_diagnostic import score_math_response
 p,d=map(Path,task);digest=sha(p)
 if d.exists():assert load(d)['source_sha']==digest;return
 b=load(p);rs=[]
 for r in b['outputs']:
  ok,pred,method=score_math_response(r['response'],b['solution'],20)
  rs.append({'replicate_index':r['replicate_index'],'correct':bool(ok),'predicted_answer':pred,'score_method':method,'tokens':len(r['response_token_ids']),'finish_reason':r['finish_reason'],'think_tags':r['response'].count('<think>')+r['response'].count('</think>')})
 atomic(d,{'source_sha':digest,'dataset_index':b['dataset_index'],'problem_id':b['problem_id'],'outputs':rs})
def score(out,rank,workers):
 c=config(out);assert c['scorer_sha']==sha(ROOT/'opd_scripts/run_opd_logit_diagnostic.py')
 assert (out/'GENERATION_COMPLETE.json').exists()
 tasks=[(str(bundle(out,i,g)),str(out/'scores'/f'{i:04d}_{g}.json')) for i in range(1000) if i%3==rank for g in range(7)]
 with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
  fs=[pool.submit(score_one,t) for t in tasks]
  for k,f in enumerate(as_completed(fs),1):
   f.result()
   if k%50==0:print('SCORE_PROGRESS',rank,k,len(tasks),flush=True)
 atomic(out/f'SCORE_RANK{rank}.json',{'tasks':len(tasks),'job':os.environ.get('SLURM_JOB_ID')})
def merge(out):
 config(out);perq=[];total=0;tokens=0;errors=0;stops=0;tags=0
 for i in range(1000):
  old=load(out/'old8'/f'{i:04d}.json');rs=[]
  for r in old:rs.append({'replicate_index':r['replicate_index'],'correct':bool(r['correct']),'score_method':r['score_method'],'tokens':len(r['response_token_ids']),'finish_reason':r['finish_reason'],'think_tags':r['response'].count('<think>')+r['response'].count('</think>')})
  for g in range(7):
   p=bundle(out,i,g);s=load(out/'scores'/f'{i:04d}_{g}.json');assert s['source_sha']==sha(p);assert s['problem_id']==old[0]['problem_id'];rs+=s['outputs']
  assert [r['replicate_index'] for r in rs]==list(range(64))
  correct=sum(r['correct'] for r in rs);total+=correct;tokens+=sum(r['tokens'] for r in rs);errors+=sum(r['score_method'] in ('timeout','error') for r in rs);stops+=sum(r['finish_reason']=='length' for r in rs);tags+=sum(r['think_tags'] for r in rs)
  perq.append({'dataset_index':i,'problem_id':old[0]['problem_id'],'correct':correct,'correct_by_replicate':[r['correct'] for r in rs]})
 summary={'questions':1000,'rollouts':64000,'reused_rollouts':8000,'new_rollouts':56000,'correct':total,'accuracy':total/64000,'pass_at_64':sum(r['correct']>0 for r in perq)/1000,'all64':sum(r['correct']==64 for r in perq)/1000,'mean_tokens':tokens/64000,'score_errors':errors,'length_stops':stops,'think_tags':tags,'correct_histogram':dict(Counter(r['correct'] for r in perq))}
 atomic(out/'per_question.json',perq);atomic(out/'summary.json',summary);atomic(out/'COMPLETE.json',{'rows':64000});print(json.dumps(summary),flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode');ap.add_argument('--out',type=Path,required=True);ap.add_argument('--rank',type=int,default=0);ap.add_argument('--shards',type=int,default=24);ap.add_argument('--workers',type=int,default=80);a=ap.parse_args()
 if a.mode=='prepare':prepare(a.out)
 elif a.mode=='generate':generate(a.out,a.rank,a.shards)
 elif a.mode=='verify':verify(a.out)
 elif a.mode=='score':score(a.out,a.rank,a.workers)
 elif a.mode=='merge':merge(a.out)
 else:raise ValueError(a.mode)
if __name__=='__main__':main()
