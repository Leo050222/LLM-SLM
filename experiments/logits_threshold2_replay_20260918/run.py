import os,json,random,hashlib,platform,math,time
from pathlib import Path
import torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo')
O=R/'opd_profile/logits_threshold2_replay_20260918'/os.environ['SLURM_JOB_ID']
O.mkdir(parents=True,exist_ok=True)
S='/storage/yukaichengLab/share/model/Qwen3-1.7B'
T='/storage/yukaichengLab/share/model/Qwen3-14B'
SEED=20260918
rows=[json.loads(x) for x in (R/'opd_eval_code/code_reasoning_grid_20260913/prompts.jsonl').read_text().splitlines()]
rows=[x for x in rows if x['dataset']=='cruxeval_i']
rng=random.Random(SEED)
first=rng.choice(rows); rows.remove(first); rng.shuffle(rows);rows.insert(0,first)
tok=AutoTokenizer.from_pretrained(S,local_files_only=True)
teacher_tok=AutoTokenizer.from_pretrained(T,local_files_only=True)
assert tok.get_vocab()==teacher_tok.get_vocab()
torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
torch.backends.cuda.matmul.allow_tf32=False
student=AutoModelForCausalLM.from_pretrained(S,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
teacher=AutoModelForCausalLM.from_pretrained(T,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()

def fw(model,ids,cache=None,keep=1):
    x=torch.tensor([ids],dtype=torch.long,device='cuda')
    n=len(ids)+(cache.get_seq_length() if cache is not None else 0)
    return model(input_ids=x,attention_mask=torch.ones((1,n),dtype=torch.long,device='cuda'),past_key_values=cache,use_cache=True,logits_to_keep=keep)

def digest(ids):return hashlib.sha256(json.dumps(ids).encode()).hexdigest()
def compare(a,b):
    af,bf=a.double(),b.double();d=(af-bf).abs();p,q=af.softmax(-1),bf.softmax(-1)
    la,lb=af.log_softmax(-1),bf.log_softmax(-1)
    ia=torch.argsort(af,descending=True,stable=True).tolist();ib=torch.argsort(bf,descending=True,stable=True).tolist()
    out=dict(exact_equal=torch.equal(a,b),bitwise_equal=torch.equal(a.view(torch.uint8),b.view(torch.uint8)),vocab_size=a.numel(),different_elements=int((a!=b).sum()),max_abs_diff=float(d.max()),mean_abs_diff=float(d.mean()),rms_diff=float(d.square().mean().sqrt()),argmax_equal=ia[0]==ib[0],total_variation=float((p-q).abs().sum()/2),kl_original_to_replay=float((p*(la-lb)).sum()),max_probability_diff=float((p-q).abs().max()),max_logprob_diff=float((la-lb).abs().max()))
    for k in [8,16]:
        out[f'top{k}']=dict(same_set=set(ia[:k])==set(ib[:k]),same_order=ia[:k]==ib[:k],overlap=len(set(ia[:k])&set(ib[:k])),original_ids=ia[:k],replay_ids=ib[:k],original_cutoff_ties=[i for i in ia if af[i]==af[ia[k-1]]],replay_cutoff_ties=[i for i in ib if bf[i]==bf[ib[k-1]]])
    out['top16_probabilities']=[dict(original_rank=j+1,token_id=i,text=tok.decode([i]),original_logit=float(af[i]),replay_logit=float(bf[i]),original_probability=float(p[i]),replay_probability=float(q[i]),relative_probability_change=float(torch.expm1(lb[i]-la[i])),logprob_change=float(lb[i]-la[i])) for j,i in enumerate(ia[:16])]
    return out

history=[];anchor=None
with torch.inference_mode():
 for attempt,row in enumerate(rows):
    prompt=row['prompt_token_ids']; generated=[];audit=[]
    assert prompt==tok.apply_chat_template([{'role':'user','content':row['problem']}],enable_thinking=False,add_generation_prompt=True,tokenize=True,return_dict=False)
    step=fw(student,prompt);cache=step.past_key_values; logits=step.logits[0,-1]
    limit=min(40264,40960-len(prompt)-1)
    while len(generated)<limit:
        base=prompt+generated;draft=[];saved=[];slps=[]
        for _ in range(min(64,limit-len(generated))):
            v=logits.detach().cpu().clone()
            token=int(torch.multinomial(logits.float().softmax(-1),1))
            saved.append(v);draft.append(token);slps.append(float(logits.float().log_softmax(-1)[token]))
            step=fw(student,[token],cache);cache=step.past_key_values;logits=step.logits[0,-1]
            if token==tok.eos_token_id:break
        # Teacher-forced full-prefix scores, aligned to each student's token.
        scored=fw(teacher,base+draft[:-1],keep=len(draft))
        tl=scored.logits[0].float().log_softmax(-1)
        tlps=tl.gather(1,torch.tensor(draft,device='cuda').view(-1,1)).flatten().tolist()
        del scored,tl
        for j,(token,slp,tlp) in enumerate(zip(draft,slps,tlps)):
            entry=dict(response_position=len(generated)+j,token_id=token,student_logprob=slp,teacher_logprob=tlp,signed_gap=slp-tlp)
            audit.append(entry)
            if slp-tlp>2:
                prefix=base+draft[:j];original=saved[j]
                anchor=dict(**entry,chunk_offset=len(generated),chunk_position=j,chunk_length=len(draft))
                break
        if anchor is not None:break
        generated+=draft
        if draft[-1]==tok.eos_token_id:break
    history.append(dict(problem_id=row['problem_id'],positions_checked=len(audit),found=anchor is not None))
    (O/'search_history.json').write_text(json.dumps(history,indent=2))
    if anchor is not None:break
    print('NO_ANCHOR',history[-1],flush=True)
    del step,cache,logits
 assert anchor is not None,'No threshold>2 position found in the dataset'
 assert all(x['signed_gap']<=2 for x in audit[:-1])
 assert prefix==prompt+(generated+draft[:j])
 assert len(prefix)==len(prompt)+anchor['response_position']
 (O/'prefix_token_ids.json').write_text(json.dumps(prefix))
 (O/'sample.json').write_text(json.dumps(row,ensure_ascii=False,indent=2))
 (O/'threshold_trace.json').write_text(json.dumps(audit,indent=2))
 replay_ids=json.loads((O/'prefix_token_ids.json').read_text());assert replay_ids==prefix
 replay=fw(student,replay_ids).logits[0,-1].detach().cpu().clone()
 repeat=fw(student,replay_ids).logits[0,-1].detach().cpu().clone()
 control=fw(student,prompt)
 for t in prefix[len(prompt):]:control=fw(student,[t],control.past_key_values)
 control_logits=control.logits[0,-1].detach().cpu().clone()
 # Hold original teacher score fixed to assess a student-only threshold flip.
 token=anchor['token_id']
 replay_lp=float(replay.float().log_softmax(-1)[token])
 result=dict(status='complete',job_id=os.environ['SLURM_JOB_ID'],node=platform.node(),gpu=torch.cuda.get_device_name(),student=S,teacher=T,backend='transformers_sdpa',dtype='bfloat16',torch=torch.__version__,transformers=transformers.__version__,seed=SEED,thinking=False,temperature=1,top_p=1,top_k=None,threshold_rule='student_chosen_logprob_minus_teacher_chosen_logprob_strictly_gt_2',teacher_scoring='full_prefix_teacher_forcing_in_64_token_chunks',student_scoring='raw_incremental_generation_logprobs',problem_id=row['problem_id'],prompt_length=len(prompt),prefix_length=len(prefix),generated_prefix_tokens=len(prefix)-len(prompt),response_prefix_text=tok.decode(prefix[len(prompt):]),prefix_sha256=digest(prefix),anchor=dict(**anchor,token_text=tok.decode([token]),replay_student_logprob=replay_lp,replay_gap_fixed_teacher=replay_lp-anchor['teacher_logprob'],replay_still_exceeds_threshold=replay_lp-anchor['teacher_logprob']>2),comparison=compare(original,replay),prefill_repeat_equal=torch.equal(replay,repeat),incremental_repeat_equal=torch.equal(original,control_logits))
 for name,v in dict(original_decode=original,fresh_prefill=replay,fresh_prefill_repeat=repeat,incremental_repeat=control_logits).items():
    assert v.numel()==151936 and torch.isfinite(v).all()
    torch.save(v,O/(name+'.pt'))
 (O/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
