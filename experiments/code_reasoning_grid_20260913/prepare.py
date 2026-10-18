import gzip, json, hashlib, subprocess
from pathlib import Path
from collections import Counter
from datasets import Dataset
from transformers import AutoTokenizer
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo')
P=R/'opd_eval_code/code_reasoning_grid_20260913'
D=P/'data'
T=AutoTokenizer.from_pretrained('/storage/yukaichengLab/share/model/Qwen3-1.7B',local_files_only=True)
rows=[]
def add(ds, pid, problem):
    ids=T.apply_chat_template([{'role':'user','content':problem}],enable_thinking=False,add_generation_prompt=True,tokenize=True,return_dict=False)
    assert 0<len(ids)<40000
    rows.append(dict(dataset=ds,problem_id=pid,dataset_index=len(rows),problem=problem,prompt_token_ids=ids))
crux=(R/'eval_repos/cruxeval/data/cruxeval.jsonl').read_bytes()
(D/'cruxeval.jsonl').write_bytes(crux)
cr=[json.loads(l) for l in crux.splitlines()]
assert len(cr)==800
for ds in ['cruxeval_o','cruxeval_i']:
    for x in cr:
        if ds.endswith('_o'):
            prompt='Given the following Python function and input, predict the output. Return only the Python value between [OUTPUT] and [/OUTPUT].\n\n'+x['code']+'\n\nf('+x['input']+')'
        else:
            prompt='Given the following Python function and desired output, provide valid arguments that produce this output. Return only the arguments to f (without f or outer call parentheses) between [INPUT] and [/INPUT].\n\n'+x['code']+'\n\nDesired output: '+x['output']
        add(ds,x['id'],prompt)
for name,ds,n in [('HumanEvalPlus','humaneval_plus',164),('MbppPlus','mbpp_plus',378)]:
    raw=gzip.decompress((D/(name+'.jsonl.gz')).read_bytes())
    (D/(name+'.jsonl')).write_bytes(raw)
    items=[json.loads(l) for l in raw.splitlines()]
    assert len(items)==n
    for x in items:
        prompt='Write a complete Python solution to the following task. Include all required imports and the full function implementation. Return only the solution in a Python code block.\n\n'+x['prompt']
        add(ds,x['task_id'],prompt)
lcb=list(Dataset.from_file(str(Path.home()/'.cache/huggingface/datasets/v6/default/0.0.0/fe58a50e77deb890/v6-test.arrow')))
assert len(lcb)==175
with (D/'livecodebench_v6_incremental.jsonl').open('w') as f:
    for x in lcb: f.write(json.dumps(x)+'\n')
for x in lcb:
    prompt='You are an expert Python programmer. Write a correct Python program that passes all tests.\n\n### Question:\n'+x['question_content']+'\n\n### Format:\n'
    if x['starter_code']:
        prompt+='Use this starter code and return the complete solution in a Python code block:\n```python\n'+x['starter_code']+'\n```'
    else:
        prompt+='Read inputs from stdin and write the answer to stdout. Return the complete solution in a Python code block.'
    add('livecodebench',x['platform']+'/'+x['question_id'],prompt)
assert len({(x['dataset'],x['problem_id']) for x in rows})==len(rows)==2317
configs=[dict(name='student_only',role='student'),dict(name='teacher_only',role='teacher')]+[dict(name=f'k{k}_t{t:g}',k=k,threshold=t) for t in [2,2.5,3,3.5,4] for k in [1,4,8,16,32]]
(P/'prompts.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
(P/'configs.json').write_text(json.dumps(configs,indent=2))
manifest=dict(counts=dict(Counter(x['dataset'] for x in rows)),total_rows=len(rows),configurations=27,total_rollouts=27*len(rows),thinking=False,temperature=1,top_p=1,versions={'humaneval_plus':'v0.1.10','mbpp_plus':'v0.2.0','livecodebench':'incremental v6 175 (not cumulative release_v6)'},lcb_dates=[min(x['contest_date'] for x in lcb),max(x['contest_date'] for x in lcb)],sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in D.glob('*')},repos={r:subprocess.check_output(['git','-C',str(R/'eval_repos'/r),'rev-parse','HEAD'],text=True).strip() for r in ['cruxeval','evalplus','LiveCodeBench']})
(P/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
