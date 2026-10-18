import os, json, random, hashlib, platform, subprocess
from pathlib import Path
import torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path('/storage/yukaichengLab/wangxin/test_eiloeo')
OUT = ROOT / 'opd_profile/logits_replay_20260918' / os.environ['SLURM_JOB_ID']
OUT.mkdir(parents=True, exist_ok=True)
MODEL = '/storage/yukaichengLab/share/model/Qwen3-1.7B'
SEED = 20260918
source = ROOT / 'opd_eval_code/code_reasoning_grid_20260913/prompts.jsonl'
rows = [json.loads(s) for s in source.read_text().splitlines()]
rows = [x for x in rows if x['dataset'] == 'cruxeval_i']
row = random.Random(SEED).choice(rows)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cuda.matmul.allow_tf32 = False
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
ids = row['prompt_token_ids']
assert ids == tok.apply_chat_template([{'role':'user','content':row['problem']}], enable_thinking=False, add_generation_prompt=True, tokenize=True, return_dict=False)
model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
prompt = torch.tensor([ids], dtype=torch.long, device='cuda')

def forward(x, cache=None):
    total = x.shape[1] + (cache.get_seq_length() if cache is not None else 0)
    return model(input_ids=x, attention_mask=torch.ones((1,total),device='cuda',dtype=torch.long), past_key_values=cache, use_cache=True, logits_to_keep=1)

with torch.inference_mode():
    initial = forward(prompt)
    cache = initial.past_key_values
    logits = initial.logits[0,-1]
    generated = []
    # Generate exactly five tokens. Each selected token is fed back, so after
    # iteration five logits predicts token SIX (not token five).
    for _ in range(5):
        token = torch.multinomial(torch.softmax(logits.float(), dim=-1), 1)
        generated.append(token.item())
        step = forward(token.view(1,1), cache)
        cache, logits = step.past_key_values, step.logits[0,-1]
    original = logits.detach().cpu().clone()
    prefix = ids + generated
    assert cache.get_seq_length() == len(prefix)
    (OUT/'prefix_token_ids.json').write_text(json.dumps(prefix))
    # Read the serialized token IDs back directly: no decoding/re-tokenizing.
    replay_ids = json.loads((OUT/'prefix_token_ids.json').read_text())
    assert replay_ids == prefix
    replay_input = torch.tensor([replay_ids],device='cuda',dtype=torch.long)
    replay = forward(replay_input).logits[0,-1].detach().cpu().clone()
    repeat = forward(replay_input).logits[0,-1].detach().cpu().clone()
    # Same incremental shape, teacher-forcing the same five sampled token IDs.
    control = forward(prompt)
    for token_id in generated:
        control = forward(torch.tensor([[token_id]],device='cuda'), control.past_key_values)
    incremental_repeat = control.logits[0,-1].detach().cpu().clone()

vectors = dict(original_decode=original, fresh_prefill=replay, fresh_prefill_repeat=repeat, incremental_repeat=incremental_repeat)
for name, vector in vectors.items():
    assert vector.numel() == model.config.vocab_size and torch.isfinite(vector).all()
    torch.save(vector, OUT/f'{name}.pt')

def compare(a,b):
    d=(a.float()-b.float()).abs()
    ia,ib=int(a.argmax()),int(b.argmax())
    return dict(exact_equal=torch.equal(a,b),bitwise_equal=torch.equal(a.view(torch.uint8),b.view(torch.uint8)),vocab_size=a.numel(),different_elements=int((a!=b).sum()),max_abs_diff=float(d.max()),mean_abs_diff=float(d.mean()),rms_diff=float(d.square().mean().sqrt()),argmax_original=ia,argmax_replay=ib,argmax_equal=ia==ib,top1_original=tok.decode([ia]),top1_replay=tok.decode([ib]),max_probability_diff=float((a.float().softmax(-1)-b.float().softmax(-1)).abs().max()))

result=dict(status='complete',job_id=os.environ['SLURM_JOB_ID'],node=platform.node(),gpu=torch.cuda.get_device_name(),cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),model=MODEL,dtype=str(original.dtype),backend='transformers',attention='sdpa',torch_version=torch.__version__,transformers_version=transformers.__version__,cuda_version=torch.version.cuda,seed=SEED,thinking=False,sampling=dict(temperature=1,top_p=1,top_k=None),problem_id=row['problem_id'],prompt_length=len(ids),prefix_length=len(prefix),generated_token_ids=generated,generated_text=tok.decode(generated),generated_tokens=[tok.decode([i]) for i in generated],eos_in_five_tokens=tok.eos_token_id in generated,prompt_token_ids=ids,prefix_sha256=hashlib.sha256(json.dumps(prefix).encode()).hexdigest(),comparison=compare(original,replay),prefill_repeat_control=compare(replay,repeat),incremental_repeat_control=compare(original,incremental_repeat))
(OUT/'sample.json').write_text(json.dumps(row,ensure_ascii=False,indent=2))
(OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
