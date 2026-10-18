import sys,json,random,hashlib,subprocess
from pathlib import Path
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');O=R/'opd_eval_hard/hard_three_20260911';D=R/'opd_eval_hard/modelscope_cache';BBH=R/'opd_eval_hard/bbh_official_20260911'
sys.path.insert(0,str(R/'eval_repos/evalscope'))
from evalscope.benchmarks.bbh.bbh_adapter import SUBSET_LIST,MULTIPLE_CHOICE_LIST,PROMPT_TEMPLATE as BBH_PROMPT
from evalscope.benchmarks.aime.aime_adapter import PROMPT_TEMPLATE as AIME_PROMPT
from evalscope.benchmarks.olympiad_bench.utils import OlympiadBenchPrompter
from transformers import AutoTokenizer
import pyarrow.parquet as pq
O.mkdir(parents=True,exist_ok=True);tokenizer=AutoTokenizer.from_pretrained('/storage/yukaichengLab/share/model/Qwen3-1.7B',local_files_only=True);rows=[];sources=[];selected={}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def add(dataset,subset,pid,problem,target,prompt,meta=None):
 ids=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
 if hasattr(ids,'keys'):ids=ids['input_ids']
 assert len(ids)<40959,(dataset,pid,len(ids))
 rows.append(dict(dataset_index=len(rows),dataset=dataset,subset=subset,problem_id=f'{dataset}/{subset}/{pid}',problem=problem,solution=str(target),prompt_token_ids=list(ids),metadata=meta or {}))
for name in ['aime24','aime25']:
 p=D/f'evalscope/{name}'/('test-00000-of-00001.parquet' if name=='aime24' else 'test.jsonl');rs=pq.read_table(p).to_pylist() if p.suffix=='.parquet' else [json.loads(l) for l in p.open()];assert len(rs)==30;sources.append({'path':str(p),'sha256':digest(p)})
 for i,r in enumerate(rs):add(name,'default',r.get('id',i),r['problem'],r.get('answer',r.get('solution')),AIME_PROMPT.format(question=r['problem']))
p=D/'AI-ModelScope/OlympiadBench/OlympiadBench/OE_TO_maths_en_COMP/OE_TO_maths_en_COMP.parquet';rs=pq.read_table(p).to_pylist();assert len(rs)==674;sources.append({'path':str(p),'sha256':digest(p)})
for r in rs:
 meta={k:r.get(k) for k in ['final_answer','answer_type','error','language','question_type','subject','is_multiple_answer','unit']}
 prompt=OlympiadBenchPrompter().make_prompt(problem=r['question'],language=r.get('language','English'),subject=r.get('subject','Math'),question_type=r.get('question_type',''),answer_type=r.get('answer_type',''),is_multiple_answer=r.get('is_multiple_answer',False),unit=r.get('unit',''))
 add('olympiad_bench','OE_TO_maths_en_COMP',r['id'],r['question'],','.join(r['final_answer']),prompt,meta)
assert len(SUBSET_LIST)==27
for name in sorted(SUBSET_LIST):
 p=BBH/'bbh'/f'{name}.json';rs=json.loads(p.read_text())['examples'];assert len(rs)>=40
 inds=sorted(random.Random('20260911:'+name).sample(range(len(rs)),40));selected[name]=inds;sources.append({'path':str(p),'sha256':digest(p)})
 for i in inds:
  r=rs[i];add('bbh',name,i,r['input'],(r['target'].strip().strip('()') if name in MULTIPLE_CHOICE_LIST else r['target'].strip()),BBH_PROMPT.format(question=r['input']),{'multiple_choice':name in MULTIPLE_CHOICE_LIST,'source_index':i})
assert len(rows)==1814 and len({r['problem_id'] for r in rows})==1814
content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows);p=O/'prompts.jsonl'
if p.exists():assert p.read_text()==content
else:p.write_text(content)
(O/'data_manifest.json').write_text(json.dumps({'rows':1814,'sources':sources,'bbh_commit':subprocess.check_output(['git','-C',str(BBH),'rev-parse','HEAD'],text=True).strip(),'bbh_selected_indices':selected,'seed':20260911,'few_shot':0,'thinking':False,'prompts_sha256':digest(p)},indent=2));print('PREPARED',len(rows),flush=True)
