import collections,hashlib,json,os,sys
from pathlib import Path
from types import SimpleNamespace
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo')
OUT=R/'opd_rescore/baseline_math_gpt4o_mini_openlux_20260916'
GRID=R/'opd_rescore/grid_math_gpt4o_mini_openlux_20260916'
sys.path.insert(0,str(R/'eval_repos/evalscope'))
from evalscope.metrics.math.parser import extract_answer
from evalscope.benchmarks.bbh.bbh_adapter import BBHAdapter
from evalscope.benchmarks.olympiad_bench.olympiad_bench_adapter import OlympiadBenchAdapter
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 assert (OUT/'system_prompt.txt').read_bytes()==(GRID/'system_prompt.txt').read_bytes()
 if (OUT/'INPUTS_COMPLETE.json').exists():
  m=json.loads((OUT/'INPUTS_COMPLETE.json').read_text());assert sha((OUT/'inputs.jsonl').read_bytes())==m['inputs_sha256'];print('INPUTS_ALREADY_VERIFIED');return
 prompts=[json.loads(l) for l in (R/'opd_eval_hard/nojudge_grid_20260912/prompts.jsonl').read_text().splitlines()]
 lookup={(x['dataset'],x['problem_id']):x for x in prompts};assert len(lookup)==2254
 sources=[('student_only','opd_eval_hard/hard_three_20260911','student',{'bbh','olympiad_bench'}),
          ('teacher_only','opd_eval_hard/hard_three_20260911','teacher',{'bbh','olympiad_bench'}),
          ('student_only','opd_eval_math500/math500_three_20260911','student',{'math500'}),
          ('teacher_only','opd_eval_hard/threshold3_mathteacher_20260911','teacher',{'math500'})]
 rows=[];seen=set();counts=collections.Counter();methods=collections.Counter()
 for config,rel,mode,allowed in sources:
  root=R/rel
  for f in sorted((root/mode/'completed').glob('*.json')):
   blob=f.read_bytes();x=json.loads(blob);ds=x.get('dataset','math500')
   if ds not in allowed:continue
   pid=x['problem_id'];canonical=pid if ds!='math500' or pid.startswith('math500/') else 'math500/'+pid
   grid=lookup[(ds,canonical)]
   assert grid['problem']==x['problem'] and grid['solution']==x['solution'],(config,pid,'problem or reference mismatch')
   assert grid['prompt_token_ids']==x['prompt_token_ids'],(config,pid,'prompt mismatch')
   assert (config,ds,canonical) not in seen;seen.add((config,ds,canonical))
   digest=sha(blob);old_path=root/'scores'/mode/f.name;old=json.loads(old_path.read_text())
   assert old['source_sha256']==digest and old['problem_id']==pid
   if old.get('status') in ['ok','missing'] and 'predicted' in old:candidate=old['predicted'];method='cached_old_predicted'
   else:
    response=x['response'].rsplit('</think>',1)[-1].strip()
    if ds=='math500':candidate=extract_answer(response)
    elif ds=='bbh':candidate=(BBHAdapter._extract_mc_answer if x['metadata']['multiple_choice'] else BBHAdapter._extract_ff_answer)(response)
    else:candidate=OlympiadBenchAdapter.extract_answer(None,response,SimpleNamespace(metadata=x['metadata']))
    method='old_extractor_replayed'
   assert candidate is None or isinstance(candidate,str)
   reference=x['metadata']['final_answer'][0] if ds=='olympiad_bench' else x['solution']
   text=json.dumps(dict(candidate_answer=candidate,reference_answer=reference),ensure_ascii=False,separators=(',',':'))
   rows.append(dict(dataset=ds,config=config,source_config=mode,problem_id=canonical,original_problem_id=pid,dataset_index=x['dataset_index'],
                    relative_path=str(f.relative_to(R)),source_sha256=digest,finish_reason=x.get('finish_reason'),
                    extraction_method=method,extraction_status='ok',candidate_answer=candidate,reference_answer=reference,
                    user_message=text,input_sha256=sha(text.encode()),old_correct=old['correct'],old_status=old['status'],old_score_path=str(old_path)))
   counts[(config,ds)]+=1;methods[method]+=1
  print('PREPARED_SOURCE',config,rel,len(rows),flush=True)
 assert len(rows)==4508
 for config in ['student_only','teacher_only']:
  for ds,n in [('math500',500),('olympiad_bench',674),('bbh',1080)]:assert counts[(config,ds)]==n
 tmp=OUT/'inputs.jsonl.tmp';tmp.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows));os.replace(tmp,OUT/'inputs.jsonl')
 m=dict(rows=len(rows),counts={g+'/'+d:n for (g,d),n in counts.items()},extraction_methods=dict(methods),
        inputs_sha256=sha((OUT/'inputs.jsonl').read_bytes()),prompt_identical_to_grid=True,all_problem_reference_and_prompt_tokens_match_grid=True)
 (OUT/'INPUTS_COMPLETE.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m),flush=True)
if __name__=='__main__':main()
