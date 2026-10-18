import argparse, collections, concurrent.futures, fcntl, hashlib, http.client, json, os, random, socket, ssl, threading, time
from pathlib import Path

OUT=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_rescore/baseline_math_gpt4o_mini_openlux_20260916')
HOST='api.openlux.ai'
MODEL='gpt-4o-mini'

def sha(b):return hashlib.sha256(b).hexdigest()
def atomic(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,path)

def validate(content):
    d=json.loads(content)
    if not isinstance(d,dict) or set(d)!={'is_correct','reason'} or type(d['is_correct']) is not bool or not isinstance(d['reason'],str):
        raise ValueError('invalid_judge_schema')
    return d

class TunnelConnection(http.client.HTTPSConnection):
    def __init__(self,port):super().__init__(HOST,timeout=120,context=ssl.create_default_context());self.port_override=port
    def connect(self):
        sock=socket.create_connection(('192.168.180.240',self.port_override),timeout=self.timeout)
        sock.sendall((OUT/'.relay_token').read_bytes())
        self.sock=self._context.wrap_socket(sock,server_hostname=HOST)

class Gate:
    def __init__(self,maximum):
        self.maximum=maximum;self.limit=min(16,maximum);self.active=0;self.good=0;self.until=0;self.cv=threading.Condition()
    def enter(self):
        with self.cv:
            while self.active>=self.limit or time.monotonic()<self.until:self.cv.wait(timeout=1)
            self.active+=1
    def leave(self,limited=False,retry_after=0,ok=False):
        with self.cv:
            self.active-=1
            if limited:
                self.limit=max(1,self.limit//2);self.good=0;self.until=max(self.until,time.monotonic()+max(2,retry_after))
            elif ok:
                self.good+=1
                if self.good>=128:self.limit=min(self.maximum,self.limit+16);self.good=0
            self.cv.notify_all()

class Client:
    def __init__(self,port,workers,prompt):
        self.port=port;self.gate=Gate(workers);self.prompt=prompt;self.local=threading.local()
        self.key=(OUT/'.openlux_api_key').read_text().strip();assert self.key.startswith('sk-')
    def call(self,user):
        body=json.dumps(dict(model=MODEL,temperature=0,max_tokens=512,stream=False,response_format={'type':'json_object'},
                             messages=[dict(role='system',content=self.prompt),dict(role='user',content=user)]),ensure_ascii=False).encode()
        history=[]
        for attempt in range(8):
            self.gate.enter();limited=False;retry_after=0;ok=False;status=None;retry=True
            try:
                conn=getattr(self.local,'conn',None)
                if conn is None:
                    conn=TunnelConnection(self.port) if self.port else http.client.HTTPSConnection(HOST,timeout=120)
                    self.local.conn=conn
                conn.request('POST','/v1/chat/completions',body,{'Authorization':'Bearer '+self.key,'Content-Type':'application/json','Accept':'application/json'})
                response=conn.getresponse();status=response.status;raw=response.read()
                if status!=200:
                    limited=status==429
                    try:retry_after=float(response.getheader('Retry-After','0'))
                    except ValueError:retry_after=0
                    retry=status in [408,409,425,429] or status>=500
                    history.append(dict(attempt=attempt+1,http_status=status,error_body_sha256=sha(raw)))
                else:
                    data=json.loads(raw);choice=data['choices'][0]
                    if choice.get('finish_reason')!='stop':raise ValueError('judge_output_not_complete')
                    decision=validate(choice['message']['content']);ok=True
                    return dict(status='ok',**decision,api_model=data.get('model'),request_id=data.get('id'),usage=data.get('usage',{}),
                                attempts=attempt+1,retry_history=history,judge_raw_content=choice['message']['content'])
            except Exception as e:
                # Never log request objects, credentials, or unsanitized exception messages.
                history.append(dict(attempt=attempt+1,error_type=type(e).__name__,http_status=status))
                conn=getattr(self.local,'conn',None)
                if conn:conn.close();self.local.conn=None
            finally:self.gate.leave(limited,retry_after,ok)
            if not retry:break
            time.sleep(max(retry_after,min(60,2**attempt))+random.random())
        return dict(status='api_error',is_correct=None,reason=None,retry_history=history,attempts=len(history))

def summarize(rows,signature):
    groups=collections.defaultdict(collections.Counter);usage=collections.Counter();pending=0
    for r in rows:
        p=OUT/'scores'/r['dataset']/r['config']/(Path(r['relative_path']).stem+'.json')
        g=groups[(r['dataset'],r['config'])];g['total']+=1
        if not p.exists():g['pending']+=1;pending+=1;continue
        s=json.loads(p.read_text());assert s['judge_signature']==signature and s['source_sha256']==r['source_sha256'] and s['input_sha256']==r['input_sha256']
        g[s['status']]+=1
        if s['is_correct'] is True:g['correct']+=1
        if isinstance(s['is_correct'],bool):g['judged']+=1
        for k,v in s.get('usage',{}).items():
            if isinstance(v,int):usage[k]+=v
    summary=[]
    for (ds,config),c in sorted(groups.items()):
        summary.append(dict(dataset=ds,config=config,**c,accuracy=c['correct']/c['judged'] if c['judged'] else None))
    errors=sum(x.get('api_error',0) for x in summary)
    result=dict(rows=len(rows),pending=pending,api_errors=errors,usage=dict(usage),groups=summary,judge_signature=signature)
    atomic(OUT/'summary.json',result)
    import csv
    fields=sorted(set().union(*(r.keys() for r in summary)))
    with (OUT/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(summary)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--tunnel-port',type=int,default=0);p.add_argument('--workers',type=int,default=128);p.add_argument('--probe',action='store_true');a=p.parse_args()
    prompt=(OUT/'system_prompt.txt').read_text();client=Client(a.tunnel_port,a.workers,prompt)
    settings=dict(provider='openlux',model=MODEL,temperature=0,max_tokens=512,response_format='json_object',prompt_sha256=sha(prompt.encode()),code_sha256=sha(Path(__file__).read_bytes()))
    signature=sha(json.dumps(settings,sort_keys=True).encode())
    if a.probe:
        tests=[('equivalent',{'candidate_answer':'2/4','reference_answer':'1/2'},True),('different',{'candidate_answer':'1','reference_answer':'100'},False),('full_text',{'candidate_answer':'At first I guessed 3. I corrected that calculation. My final answer is 4.','reference_answer':'4'},True)]
        results=[]
        for name,payload,expected in tests:
            r=client.call(json.dumps(payload));r.update(test=name,expected=expected);results.append(r)
            if r['status']!='ok' or r['is_correct']!=expected:
                atomic(OUT/'PROBE.json',dict(passed=False,results=results,settings=settings));raise SystemExit('API_PROBE_FAILED; inspect PROBE.json')
        atomic(OUT/'PROBE.json',dict(passed=True,results=results,settings=settings));print('API_PROBE_PASSED',flush=True);return
    lock=(OUT/'scoring.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    meta=json.loads((OUT/'INPUTS_COMPLETE.json').read_text());raw=(OUT/'inputs.jsonl').read_bytes();assert sha(raw)==meta['inputs_sha256']
    rows=[json.loads(line) for line in raw.splitlines()];assert len(rows)==4508
    atomic(OUT/'RUN_CONFIG.json',dict(**settings,judge_signature=signature,max_concurrency=a.workers,job_id=os.getenv('SLURM_JOB_ID'),host=socket.gethostname(),started_at=time.time()))
    def action(r):
        dest=OUT/'scores'/r['dataset']/r['config']/(Path(r['relative_path']).stem+'.json')
        if dest.exists():
            old=json.loads(dest.read_text());assert old['judge_signature']==signature and old['input_sha256']==r['input_sha256'] and old['source_sha256']==r['source_sha256']
            if old['status'] in ['ok','extraction_missing']:return old
        if r['candidate_answer'] is None or not r['candidate_answer'].strip():result=dict(status='extraction_missing',is_correct=False,reason='旧提取结果为空；本条未调用模型',attempts=0)
        else:result=client.call(r['user_message'])
        result.update({k:r[k] for k in ['dataset','config','problem_id','dataset_index','relative_path','source_sha256','input_sha256','finish_reason','extraction_method']})
        result.update(judge_signature=signature,completed_at=time.time());atomic(dest,result);return result
    # Pilot uses actual extracted values spanning all three datasets.
    pilot=[]
    for ds in ['math500','olympiad_bench','bbh']:pilot.extend([r for r in rows if r['dataset']==ds][:4])
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:pilot_results=list(pool.map(action,pilot))
    atomic(OUT/'PILOT.json',dict(rows=len(pilot_results),statuses=dict(collections.Counter(r['status'] for r in pilot_results))))
    if any(r['status']=='api_error' for r in pilot_results):raise SystemExit('PILOT_API_ERRORS; full dispatch stopped')
    print('PILOT_PASSED',len(pilot_results),flush=True)
    pending=rows
    for round_id in range(3):
        errors=[];done=0;started=time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures={pool.submit(action,r):r for r in pending}
            for future in concurrent.futures.as_completed(futures):
                r=future.result();done+=1
                if r['status']=='api_error':errors.append(futures[future])
                if done%100==0:
                    progress=dict(round=round_id+1,processed=done,round_total=len(pending),api_errors=len(errors),active_limit=client.gate.limit,elapsed_seconds=round(time.time()-started,1),updated_at=time.time())
                    atomic(OUT/'progress.json',progress);print(json.dumps(progress),flush=True)
        summary=summarize(rows,signature)
        if not errors:break
        pending=errors;time.sleep(30)
    summary=summarize(rows,signature)
    if summary['pending'] or summary['api_errors']:raise SystemExit('UNRESOLVED_API_ERRORS; resumable output preserved')
    atomic(OUT/'COMPLETE.json',dict(rows=len(rows),completed_at=time.time(),judge_signature=signature,usage=summary['usage']))
    print('SCORING_COMPLETE',len(rows),flush=True)

if __name__=='__main__':main()
