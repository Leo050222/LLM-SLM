#!/usr/bin/env python3
"""One AsyncLLM per GPU; bounded8 inflight, independent completion/refill."""
import argparse,asyncio,fcntl,sys,time,json
from pathlib import Path
sys.path.insert(0,'/storage/yukaichengLab/wangxin/test_eiloeo/opd_online_code/20260911_chunk64_k8_2nodes/ocrd_pipeline')
from online_rollout.core import atomic_json,digest
from anchor_selector.core import lookup_actual_logprob,sorted_topk,tokenizer_fingerprint
from async_service import serve

def extract(output,req,role,vocab_size):
 if not output.finished:raise ValueError('Incomplete engine output')
 if req['op']=='score':
  ids=req['prefix']+req['tokens'];entries=output.prompt_logprobs
  if entries is None or len(entries)!=len(ids):raise ValueError('Misaligned prompt logprobs')
  chosen=[];top_ids=[];top_lp=[]
  for pos in range(len(req['prefix']),len(ids)):
   chosen.append(lookup_actual_logprob(entries[pos],ids[pos]))
   ii,ll=([],[]) if role=='student' else sorted_topk(entries[pos],16,vocab_size)
   top_ids.append(ii);top_lp.append(ll)
  return dict(chosen=chosen,top_ids=top_ids,top_lp=top_lp)
 generated=output.outputs[0]
 return dict(token_ids=list(generated.token_ids),finish_reason=generated.finish_reason)

async def main(args):
 import torch,vllm
 from vllm import SamplingParams
 from vllm.sampling_params import RequestOutputKind
 from vllm.inputs import TokensPrompt
 from vllm.usage.usage_lib import UsageContext
 from vllm.engine.arg_utils import AsyncEngineArgs
 from vllm.v1.engine.async_llm import AsyncLLM
 torch.manual_seed(20260806);x=torch.randn(256,256,device='cuda');y=x@x;torch.cuda.synchronize()
 print('CUDA_WITNESS_OK',torch.cuda.get_device_name(),float(y.sum()),flush=True);del x,y
 base=args.queue/args.role;(base/'requests').mkdir(parents=True,exist_ok=True);(base/'results').mkdir(exist_ok=True)
 lock=(base/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 engine=AsyncLLM.from_engine_args(AsyncEngineArgs(model=args.model,tensor_parallel_size=1,dtype='bfloat16',max_model_len=args.max_model_len,max_num_seqs=args.batch_size,gpu_memory_utilization=args.gpu_memory_utilization,enable_prefix_caching=False,max_logprobs=16,disable_log_stats=True),usage_context=UsageContext.LLM_CLASS)
 tokenizer=engine.get_tokenizer();vocab_size=len(tokenizer)
 info=dict(model=args.model,tokenizer_sha256=tokenizer_fingerprint(tokenizer),max_model_len=args.max_model_len,protocol=3,worker_sha256=digest(Path(__file__).read_text()),vllm_version=vllm.__version__,eos_token_id=tokenizer.eos_token_id)
 if (base/'model.json').exists() and json.loads((base/'model.json').read_text())!=info:raise ValueError('Queue identity mismatch')
 atomic_json(base/'model.json',info);atomic_json(base/'ready.json',info)
 async def execute(req,key,wait):
  if req['model_identity']!=digest(info):raise ValueError('Model identity mismatch')
  ids=req['prefix']
  if req['op']=='score':
   ids=ids+req['tokens'];params=SamplingParams(max_tokens=1,temperature=1.,top_p=1.,prompt_logprobs=16,seed=0,output_kind=RequestOutputKind.FINAL_ONLY)
  elif req['op']=='generate':
   extra={}
   if req.get('banned_first') is not None:
    if req['n']!=1:raise ValueError('First-token ban requires n=1')
    extra['_bad_words_token_ids']=[[req['banned_first']]]
   params=SamplingParams(max_tokens=req['n'],temperature=1.,top_p=1.,seed=req['seed'],logprobs=1,output_kind=RequestOutputKind.FINAL_ONLY,**extra)
  else:raise ValueError('Unknown operation')
  if not ids or len(ids)+params.max_tokens>args.max_model_len:raise ValueError('Context overflow')
  started=time.monotonic();output=None
  stream=engine.generate(TokensPrompt(prompt_token_ids=ids),params,request_id=key)
  try:
   async for item in stream:output=item
  finally:
   await stream.aclose()
  if output is None:raise ValueError('Missing output')
  value=await asyncio.to_thread(extract,output,req,args.role,vocab_size)
  value.update(worker_request_seconds=time.monotonic()-started,queue_wait_seconds=wait,operation=req['op'],inflight_limit=args.batch_size)
  return value
 try:await serve(base,args.queue/'STOP',execute,args.batch_size)
 finally:engine.shutdown()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--queue',type=Path,required=True);p.add_argument('--role',choices=['student','teacher'],required=True);p.add_argument('--model',required=True);p.add_argument('--max-model-len',type=int,default=40960);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--gpu-memory-utilization',type=float,default=.9)
 asyncio.run(main(p.parse_args()))
