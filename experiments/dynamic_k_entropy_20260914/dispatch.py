import os,json,time,queue,threading,fcntl,hashlib
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from ablation_core import atomic_json,digest
from dynamic import Settings,rollout,verify_trace,finalize_timeout,paired_entropy
from backend import Backend,tokenizer_fingerprint
from deadline import Budget,TraceDeadline

ROOT=Path('/storage/yukaichengLab/wangxin/test_eiloeo')
P=ROOT/'opd_eval_dynamic/dynamic_k_entropy_20260914'

from queue_rpc import RPC
from migration import migrate_manifest

def prepare():
    for group,origin,n in [('math',ROOT/'opd_eval_hard/nojudge_grid_20260912',2254),
                           ('code',ROOT/'opd_eval_code/code_reasoning_grid_20260913',2317)]:
        data=(origin/'prompts.jsonl').read_bytes();rows=[json.loads(l) for l in data.splitlines() if l]
        assert len(rows)==n and not any(r['dataset'].startswith('aime') for r in rows)
        p=P/group/'prompts.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():assert p.read_bytes()==data
        else:p.write_bytes(data)
    print('PREPARED 4571',flush=True)

def main():
    from transformers import AutoTokenizer
    slot=Path(os.environ['OPD_ONLINE_RUN']);Q=slot/'queue'
    lock=(P/'dispatch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for _ in range(360):
        if len(list(Q.glob('replica_*/*/ready.json')))==4:break
        time.sleep(5)
    else:raise RuntimeError('Model readiness timeout')
    models=[{role:json.loads((Q/f'replica_{i}'/role/'model.json').read_text()) for role in ['student','teacher']} for i in range(2)]
    assert models[0]==models[1]
    tokenizer=AutoTokenizer.from_pretrained(models[0]['student']['model'],local_files_only=True)
    assert all(x['tokenizer_sha256']==tokenizer_fingerprint(tokenizer) for x in models[0].values())
    settings=Settings();tasks=queue.Queue();allrows={}
    for group in ['math','code']:
        rows=[json.loads(l) for l in (P/group/'prompts.jsonl').read_text().splitlines() if l];allrows[group]=rows
        manifest=dict(settings=asdict(settings),models=models[0],input_sha256=digest(rows),expected_rows=len(rows),
                      thinking=False,sampling=dict(temperature=1.,top_p=1.),judge_config=None,first_token_ban=True,
                      trace_limit_seconds=900,entropy_units='nats',entropy_scope='full unmodified vocabulary',
                      code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
        p=P/group/'manifest.json'
        migrate_manifest(p,manifest,atomic_json,digest)
        for row in rows:
            if not (P/group/'completed'/f"{row['dataset_index']:05d}.json").exists():tasks.put((group,row))
    # Real-model service smoke before formal generation, all four engines.
    witness=[]
    for i in range(2):
        b=Backend(RPC(Q/f'replica_{i}',Budget(slot/f'smoke_{i}.json',limit=600)),models[i],tokenizer)
        row=allrows['math'][i];prefix=row['prompt_token_ids']
        hs,ht=paired_entropy(b,prefix);assert 0<=hs<13 and 0<=ht<13
        first=b.generate('teacher',prefix,1,123+i)
        assert len(first['token_ids'])==1
        after=paired_entropy(b,prefix+first['token_ids'])
        witness.append(dict(replica=i,before=[hs,ht],after=after))
    atomic_json(slot/'SMOKE_PASSED.json',dict(real_entropy=witness));print('SMOKE_PASSED',flush=True)
    failures=[];guard=threading.Lock();done=0
    def action(group,row,replica):
        nonlocal done
        d=P/group;p=d/'traces'/f"{row['dataset_index']:05d}.json"
        state=json.loads(p.read_text()) if p.exists() else dict(identity=digest(dict(row=row,settings=asdict(settings))),
            response_token_ids=[],token_sources=[],events=[],rounds=0,student_draft_tokens=0,
            discarded_tokens=0,judge_calls=0,status='RUNNING')
        state.setdefault('execution_history',[]).append(dict(slot=slot.name, worker_protocol=2, resumed=bool(p.exists())))
        budget=Budget(d/'timings'/p.name);b=Backend(RPC(Q/f'replica_{replica}',budget),models[replica],tokenizer)
        def save(x):budget.persist();atomic_json(p,x)
        try:
            try:result=rollout(row,settings,b,state,save)
            except TraceDeadline:result=finalize_timeout(state)
            budget.persist();result['trace_elapsed_seconds']=budget.elapsed();verify_trace(result);save(result)
            output=dict(row,**result);output['response']=tokenizer.decode(result['response_token_ids'],skip_special_tokens=True)
            atomic_json(d/'completed'/p.name,output)
            with guard:
                done+=1;print('DONE',done,group,row['dataset_index'],result['finish_reason'],flush=True)
        except Exception as e:
            budget.persist()
            with guard:failures.append(dict(group=group,index=row['dataset_index'],error=repr(e)))
            print('ERROR',group,row['dataset_index'],repr(e),flush=True)
    def consume(replica):
        while True:
            try:group,row=tasks.get_nowait()
            except queue.Empty:return
            action(group,row,replica)
    try:
        with ThreadPoolExecutor(max_workers=64) as pool:
            fs=[pool.submit(consume,i%2) for i in range(64)]
            for f in fs:f.result()
        if failures:atomic_json(slot/'FAILED.json',failures);raise RuntimeError('Trace failures, inspect before retry')
        for group,rows in allrows.items():
            for row in rows:
                x=json.loads((P/group/'completed'/f"{row['dataset_index']:05d}.json").read_text())
                verify_trace(x);assert x['problem_id']==row['problem_id'] and x['prompt_token_ids']==row['prompt_token_ids']
            atomic_json(P/group/'VERIFIED.json',dict(rows=len(rows)))
        atomic_json(P/'ALL_GENERATION_COMPLETE.json',dict(rows=4571));print('ALL_GENERATION_COMPLETE',flush=True)
    finally:
        for i in range(2):(Q/f'replica_{i}'/'STOP').touch()

if __name__=='__main__':
    import sys
    if '--prepare' in sys.argv:prepare()
    else:main()
