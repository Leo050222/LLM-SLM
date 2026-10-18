import json,sys,os,signal,hashlib,importlib.util,collections,csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from ablation_core import atomic_json
ROOT=Path('/storage/yukaichengLab/wangxin/test_eiloeo');P=ROOT/'opd_eval_dynamic/dynamic_k_entropy_20260914'

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def action(path):
    path=Path(path);group=path.parents[1].name;dest=P/group/'scores'/path.name
    blob=path.read_bytes();sha=hashlib.sha256(blob).hexdigest();x=json.loads(blob)
    if dest.exists():
        r=json.loads(dest.read_text());assert r['source_sha256']==sha;return r
    if x['finish_reason']=='trace_timeout':result=dict(correct=False,status='trace_timeout')
    elif group=='math':
        m=module('math_scoring',ROOT/'opd_scripts/nojudge_grid_20260912/score.py')
        def timeout(*args):raise TimeoutError()
        signal.signal(signal.SIGALRM,timeout)
        try:
            signal.alarm(30);correct,pred=m.score_answer(x);result=dict(correct=bool(correct),predicted=pred,status='ok' if pred else 'missing')
        except TimeoutError:result=dict(correct=False,status='timeout')
        except Exception as e:result=dict(correct=False,status=type(e).__name__)
        finally:signal.alarm(0)
    else:
        m=module('code_scoring',ROOT/'opd_scripts/code_reasoning_grid_20260913/score.py')
        ds=x['dataset'];filename,key={
          'cruxeval_o':('cruxeval.jsonl','id'),'cruxeval_i':('cruxeval.jsonl','id'),
          'humaneval_plus':('HumanEvalPlus.jsonl','task_id'),'mbpp_plus':('MbppPlus.jsonl','task_id'),
          'livecodebench':('livecodebench_v6_incremental.jsonl','question_id')}[ds]
        rows=[json.loads(l) for l in (m.P/'data'/filename).read_text().splitlines()]
        problems={(r['platform']+'/'+r[key] if ds=='livecodebench' else r[key]):r for r in rows}
        result=m.sandbox(dict(dataset=ds,problem=problems[x['problem_id']],solution=m.extract(ds,x['response'])))
    events=[e for e in x['events'] if 'bridge_token_ids' in e]
    result.update(dataset=x['dataset'],problem_id=x['problem_id'],dataset_index=x['dataset_index'],
        source_sha256=sha,tokens=len(x['response_token_ids']),teacher_tokens=x['token_sources'].count('teacher'),
        interventions=len(events),bridge_lengths=[len(e['bridge_token_ids']) for e in events])
    atomic_json(dest,result);return result

def main():
    assert json.loads((P/'ALL_GENERATION_COMPLETE.json').read_text())['rows']==4571
    files=list((P/'math/completed').glob('*.json'))+list((P/'code/completed').glob('*.json'));assert len(files)==4571
    results=[]
    with ProcessPoolExecutor(max_workers=32) as pool:
        for i,r in enumerate(pool.map(action,files,chunksize=1)):
            results.append(r)
            if i%100==0:print('SCORED',i+1,4571,flush=True)
    output=[]
    for ds in sorted({r['dataset'] for r in results}):
        rs=[r for r in results if r['dataset']==ds];n=len(rs);lengths=[k for r in rs for k in r['bridge_lengths']]
        output.append(dict(dataset=ds,n=n,correct=sum(r['correct'] for r in rs),acc=sum(r['correct'] for r in rs)/n,
          mean_interventions=sum(r['interventions'] for r in rs)/n,mean_output_tokens=sum(r['tokens'] for r in rs)/n,
          teacher_token_fraction=sum(r['teacher_tokens'] for r in rs)/sum(r['tokens'] for r in rs),
          mean_actual_k=sum(lengths)/len(lengths) if lengths else 0,k_histogram=dict(collections.Counter(lengths)),
          score_status=dict(collections.Counter(r['status'] if isinstance(r['status'],str) else json.dumps(r['status'],sort_keys=True) for r in rs))))
    atomic_json(P/'summary.json',output)
    with (P/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    atomic_json(P/'SCORING_COMPLETE.json',dict(rows=4571));print(json.dumps(output),flush=True)

if __name__=='__main__':main()
