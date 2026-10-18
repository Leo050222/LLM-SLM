import ast, collections, json, re, sys
from pathlib import Path
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo')
sys.path.insert(0,str(R/'eval_repos/evalscope'))
from evalscope.metrics.math.parser import extract_answer
from evalscope.benchmarks.bbh.bbh_adapter import BBHAdapter

stats=collections.defaultdict(collections.Counter)
examples=collections.defaultdict(list)
def flag(c, key, f, pred):
    c[key]+=1
    ek=key
    if len(examples[ek])<4: examples[ek].append(dict(path=str(f),extracted=str(pred)[:600]))

roots=[('math_grid',R/'opd_eval_hard/nojudge_grid_20260912'),
       ('code_grid',R/'opd_eval_code/code_reasoning_grid_20260913'),
       ('dynamic',R/'opd_eval_dynamic/dynamic_k_entropy_20260914'),
       ('aime_historical',R/'opd_eval_hard/hard_three_20260911')]
for experiment,root in roots:
    for folder in sorted(root.glob('*/completed')):
        group=folder.parent.name
        count=0
        for f in folder.glob('*.json'):
            x=json.loads(f.read_text());ds=x['dataset']
            if experiment=='aime_historical' and ds not in ['aime24','aime25']:continue
            c=stats[(experiment,group,ds)];c['n']+=1;count+=1
            if x.get('finish_reason')=='trace_timeout':c['trace_timeout']+=1;continue
            c['attempted']+=1
            response=x['response'];s=response.rsplit('</think>',1)[-1].strip()
            pred=''
            if ds in ['math500','aime24','aime25']:
                try: pred=extract_answer(s)
                except Exception:c['extraction_exception']+=1;continue
                if 'boxed' in s:c['boxed_branch']+=1
                elif any(t in s for t in ['he answer is','final answer is','答案是','ANSWER:']):c['answer_marker_branch']+=1
                else:flag(c,'last_number_fallback',f,pred)
            elif ds=='bbh':
                mc=x['metadata']['multiple_choice']
                pred=BBHAdapter._extract_mc_answer(s) if mc else BBHAdapter._extract_ff_answer(s)
                if 'So the answer is ' in s:c['exact_answer_marker']+=1
                else:flag(c,'bbh_no_exact_marker',f,pred)
                if mc:
                    c['mc_n']+=1
                    if re.fullmatch('[A-Z]',pred or ''):c['mc_letter']+=1
                    else:flag(c,'mc_not_letter',f,pred)
                else:
                    c['ff_n']+=1
                    sub=x['subset']; valid=True
                    if sub in ['navigate','sports_understanding','causal_judgement','web_of_lies']:valid=pred.casefold() in ['yes','no']
                    elif sub=='boolean_expressions':valid=pred.casefold() in ['true','false']
                    elif sub=='formal_fallacies':valid=pred.casefold() in ['valid','invalid']
                    elif sub in ['object_counting','multistep_arithmetic_two']:valid=bool(re.fullmatch(r'-?\d+',pred.strip()))
                    elif sub=='dyck_languages':valid=bool(re.fullmatch(r'[\s()\[\]{}<>]+',pred))
                    elif sub=='word_sorting':valid=bool(re.fullmatch(r'[A-Za-z]+(?:\s+[A-Za-z]+)*',pred))
                    if valid:c['ff_shape_valid']+=1
                    else:flag(c,'ff_shape_suspicious',f,pred)
            elif ds=='olympiad_bench':
                marker='所以最终答案是' if x['metadata']['language']=='Chinese' else 'So the final answer is '
                hits=re.findall(re.escape(marker)+'(.*)',s)
                pred=hits[-1].strip() if hits else s
                if hits:c['exact_final_marker']+=1
                boxes=list(re.finditer(r'\\boxed{',pred))
                if boxes:
                    c['has_boxed']+=1
                    if len(boxes)>1:flag(c,'multiple_boxes',f,pred)
                    malformed=False
                    for m in boxes:
                        depth=1
                        for ch in pred[m.end():]:
                            depth+=(ch=='{')-(ch=='}')
                            if depth==0:break
                        if depth:malformed=True
                    if malformed:flag(c,'malformed_box',f,pred)
                elif re.findall(r'\$(.*?)\$',pred.strip().split('\n')[-1]):c['last_line_math']+=1
                else:flag(c,'whole_text_fallback',f,pred)
            elif ds.startswith('cruxeval'):
                tag='INPUT' if ds.endswith('_i') else 'OUTPUT'
                hits=re.findall(r'\['+tag+r'\](.*?)\[/'+tag+r'\]',response,re.S)
                pred=hits[-1].strip() if hits else response.strip()
                if hits:c['tag_match']+=1
                else:flag(c,'no_tag_fallback',f,pred)
                try:ast.parse('f('+pred+')' if ds.endswith('_i') else pred,mode='eval');c['syntax_valid']+=1
                except (SyntaxError,ValueError,RecursionError):flag(c,'syntax_invalid',f,pred)
            else:
                blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',response,re.S)
                pred=blocks[0].strip() if blocks else response.strip()
                if blocks:c['code_block_match']+=1
                else:flag(c,'no_code_block_fallback',f,pred)
                if len(blocks)>1:flag(c,'multiple_code_blocks',f,pred)
                try:ast.parse(pred);c['syntax_valid']+=1
                except (SyntaxError,ValueError,RecursionError):flag(c,'syntax_invalid',f,pred)
            if pred and pred.strip():c['nonempty']+=1
            else:flag(c,'empty',f,pred)
        print(experiment,group,count,file=sys.stderr,flush=True)
print(json.dumps(dict(rows=[dict(experiment=e,group=g,dataset=d,**c) for (e,g,d),c in stats.items()],examples=examples),ensure_ascii=False))
