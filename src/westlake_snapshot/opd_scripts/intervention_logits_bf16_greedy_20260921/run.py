"""Freeze dual-greedy main-method events and reuse validated BF16 replay engine."""
import argparse,collections,json,os,shutil,time
from pathlib import Path
from replay_base import (ROOT,MODELS,PROTOCOL,VOCAB,sha,read,encoded,atomic_json,file_lock,
                         freeze_trace,chunk_selected,load_manifest,score,verify)

OUT=ROOT/'opd_analysis/intervention_logits_bf16_greedy_20260921'
SOURCE=ROOT/'opd_eval_ablation/main_greedy_k1_t2_20260921'
GROUP=SOURCE/'r1_main_greedy_k1_t2'
EXPECTED={'cruxeval_o':800,'cruxeval_i':800,'humaneval_plus':164,'mbpp_plus':378,
          'livecodebench':175,'olympiad_bench':674,'bbh':1080,'math500':500}

def prepare(out):
    lock=file_lock(out/'prepare.lock')
    if (out/'INPUTS_COMPLETE.json').exists():
        load_manifest(out);print(json.dumps(read(out/'INPUTS_COMPLETE.json')),flush=True);return
    generation=read(SOURCE/'ALL_GENERATION_COMPLETE.json');verified=read(GROUP/'VERIFIED.json')
    assert generation['rows']==verified['rows']==sum(EXPECTED.values())==4571
    m=read(GROUP/'manifest.json')
    assert m['sampling']==dict(temperature=0.,top_p=1.)
    assert m['settings']['teacher_tokens']==1 and m['settings']['gap_threshold']==2
    assert m['thinking'] is False and m['first_token_ban'] is True
    for role in MODELS:assert m['models'][role]['model']==str(MODELS[role])
    inputs=out/'inputs';inputs.mkdir(exist_ok=True)
    files=sorted((GROUP/'completed').glob('*.json'));assert len(files)==4571
    pending=[];ne=0;ntokens=0;shards=[];event_count=0;total_tokens=0
    counts=collections.defaultdict(collections.Counter);keys=set();smoke={};longest=None;noevents=[];timeouts=0
    def flush():
        nonlocal pending,ne,ntokens
        if not pending:return
        name=f'{len(shards):05d}.jsonl';p=inputs/name;tmp=p.with_suffix('.tmp')
        with tmp.open('wb') as f:
            for row in pending:f.write(encoded(row)+b'\n')
            f.flush();os.fsync(f.fileno())
        tmp.replace(p)
        shards.append(dict(name=name,sha256=sha(p),traces=len(pending),events=ne,input_tokens=ntokens,
                           first_event_id=pending[0]['events'][0]['event_id']))
        pending=[];ne=0;ntokens=0
    for p in files:
        x=read(p);ds=x['dataset'];assert ds in EXPECTED
        key=(ds,x['problem_id']);assert key not in keys;keys.add(key)
        family='math' if ds in ['math500','bbh','olympiad_bench'] else 'code'
        row=freeze_trace(x,p,0,family,event_count);n=len(row['events']);event_count+=n
        counts[ds]['traces']+=1;counts[ds]['events']+=n
        if row['finish_reason']=='trace_timeout':timeouts+=1;counts[ds]['timeout_traces']+=1
        if not n:
            noevents.append({k:v for k,v in row.items() if k not in ['input_ids','events']});continue
        if ds not in smoke:
            s=dict(row);s['events']=row['events'][:1];s['input_ids']=row['input_ids'][:s['events'][0]['prefix_length']];smoke[ds]=s
        if longest is None or len(row['input_ids'])>len(longest['input_ids']):longest=row
        if pending and (ne+n>1024 or ntokens+len(row['input_ids'])>250000):flush()
        pending.append(row);ne+=n;ntokens+=len(row['input_ids']);total_tokens+=len(row['input_ids'])
    flush()
    assert {d:c['traces'] for d,c in counts.items()}==EXPECTED
    assert timeouts==verified['timeout_rows'] and len(smoke)==8
    bytes_needed=event_count*VOCAB*4
    assert shutil.disk_usage(out).free>bytes_needed*1.2+10_000_000_000
    atomic_json(out/'no_intervention_traces.json',noevents)
    atomic_json(out/'smoke_inputs.json',list(smoke.values()));atomic_json(out/'longest_trace.json',longest)
    manifest=dict(protocol=PROTOCOL,dtype='BF16',vocabulary_size=VOCAB,models={k:str(v) for k,v in MODELS.items()},
        traces=4571,events=event_count,no_intervention_traces=len(noevents),timeout_traces=timeouts,
        forward_input_tokens_per_model=total_tokens,tensor_bytes_two_models=bytes_needed,
        historical_sampling=m['sampling'],raw_logits_temperature='unscaled; softmax T=1 for preference analysis',
        groups=[dict(source=str(GROUP),replicate=0,manifest_sha256=sha(GROUP/'manifest.json'),manifest=m)],
        datasets=dict(counts),shards=shards,freeze_script_sha256=sha(__file__),
        generation_receipt_sha256=sha(SOURCE/'ALL_GENERATION_COMPLETE.json'),source_verified_sha256=sha(GROUP/'VERIFIED.json'),
        smoke_sha256=sha(out/'smoke_inputs.json'),longest_sha256=sha(out/'longest_trace.json'),
        no_intervention_sha256=sha(out/'no_intervention_traces.json'))
    atomic_json(out/'manifest.json',manifest)
    atomic_json(out/'INPUTS_COMPLETE.json',dict(manifest_sha256=sha(out/'manifest.json'),traces=4571,events=event_count,
        shards=len(shards),no_intervention_traces=len(noevents),timeout_traces=timeouts,input_tokens_per_model=total_tokens,
        tensor_bytes_two_models=bytes_needed,created=time.time()))
    print(json.dumps(read(out/'INPUTS_COMPLETE.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','score','verify']);p.add_argument('--role',choices=list(MODELS));a=p.parse_args()
    if a.command=='prepare':prepare(OUT)
    elif a.command=='verify':verify(OUT)
    else:assert a.role;score(OUT,a.role)
