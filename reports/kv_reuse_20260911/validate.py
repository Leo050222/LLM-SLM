import json,os,sys,time
from pathlib import Path
sys.path.insert(0,os.environ['OPD_CODE_ROOT']+'/ocrd_pipeline')
from online_rollout.core import FileRPC,digest,atomic_json
root=Path(os.environ['OPD_KV_RUN'])
models={m:{r:json.loads((root/m/r/'model.json').read_text()) for r in ['student','teacher']} for m in ['cached','reference']}
rpcs={m:FileRPC(root/m,timeout=600) for m in models}
metrics=[]
def call(mode,role,**req):
 out=rpcs[mode].call(role,dict(model_identity=digest(models[mode][role]),**req))
 metrics.append(dict(mode=mode,role=role,op=req['op'],prefix=len(req['prefix']),**{k:out.get(k) for k in ['num_cached_tokens','computed_prompt_tokens','worker_batch_seconds','logprob_sidecar_hit','boundary_repairs','boundary_repair_seconds']}))
 return out

def check(prefix,tokens):
 for role in ['student','teacher']:
  a=call('cached',role,op='score',prefix=prefix,tokens=tokens)
  b=call('reference',role,op='score',prefix=prefix,tokens=tokens)
  error=max(abs(x-y) for x,y in zip(a['chosen'],b['chosen']))
  assert len(a['chosen'])==len(b['chosen'])==len(tokens)
  assert error<.5,(role,error)  # BF16 guard, NOT bitwise/numerical-equivalence claim
  print(json.dumps(dict(role=role,prefix=len(prefix),tokens=len(tokens),max_error=error)),flush=True)
  errors.append(error)

errors=[]
source=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/online_chunk64_k32_test1000_h100_seed20260911/prompts1000.jsonl')
row=json.loads(source.read_text().splitlines()[0]);prefix=row['prompt_token_ids']
# Warm full prefix, then append, roll back, and diverge twice.
for step in range(5):
 draft=call('cached','student',op='generate',prefix=prefix,n=64,seed=12345+step)['token_ids']
 check(prefix,draft)
 if step in [1,3]:
  rollback=prefix+draft[:17]
  entry=call('cached','teacher',op='generate',prefix=rollback,n=1,seed=900+step,banned_first=draft[17])['token_ids']
  assert entry[0]!=draft[17]
  # Score the banned entry against raw unmasked reference probabilities.
  check(rollback,entry)
  tail=call('cached','teacher',op='generate',prefix=rollback+entry,n=7,seed=901+step)['token_ids']
  prefix=rollback+entry+tail
 else:prefix+=draft
# Exact block-aligned new target: KV exists but its probability does not.
aligned=(prefix*10)[:512]
for role in ['student','teacher']:
 call('cached',role,op='generate',prefix=aligned+[90,91,92,93]*16,n=1,seed=813)
check(aligned,[80,81,82,83])
assert any(m.get('boundary_repairs') for m in metrics),'Boundary repair was not exercised'
# Stronger long-prefix witness, then append64 without recomputing history.
long_prefix=row['prompt_token_ids']+(prefix[len(row['prompt_token_ids']):]*20)[:4096]
check(long_prefix,[100,101,102,103])
check(long_prefix+[100,101,102,103],[104,105,106,107])
hits=[m for m in metrics if m['mode']=='cached' and (m.get('num_cached_tokens') or 0)>0]
assert hits,'No GPU KV cache hit'
assert any(m['logprob_sidecar_hit'] for m in metrics if m['mode']=='cached' and m['role']=='student' and m['op']=='score')
# Control: full-prefill vs decode differences even with prefix caching disabled.
control=[]
for role in ['student','teacher']:
 g=call('reference',role,op='generate',prefix=prefix,n=64,seed=814)
 s=call('reference',role,op='score',prefix=prefix,tokens=g['token_ids'])
 delta=[abs(x-y) for x,y in zip(g['raw_logprobs']['chosen'],s['chosen'])]
 control.append(dict(role=role,max_error=max(delta),mean_error=sum(delta)/len(delta)))
result=dict(status='KV_REUSE_VALIDATED_WITH_BF16_DRIFT',max_logprob_error=max(errors),strict_005_tolerance_passed=max(errors)<.05,decode_prefill_control=control,metrics=metrics)
atomic_json(root/'validation.json',result)
print(json.dumps(dict(status=result['status'],max_logprob_error=max(errors),gpu_cache_hits=len(hits),decode_prefill_control=control,cached_prefill_tokens=sum(m['computed_prompt_tokens'] for m in metrics if m['mode']=='cached'),reference_score_prefill_tokens=sum(m['computed_prompt_tokens'] for m in metrics if m['mode']=='reference'))),flush=True)
