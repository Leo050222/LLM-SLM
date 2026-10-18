"""CPU-only scoring; per-bundle checkpoints, then hierarchical summaries."""
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path
from pipeline import source,load,atomic,sha,ROOT,H,teacher_path,student_path,validate_teacher,validate_student
from run_opd_logit_diagnostic import score_math_response

def score_file(task):
    p,dest=map(Path,task)
    digest=sha(p)
    if dest.exists():
        cached=load(dest);assert cached['source_sha256']==digest
        return str(dest)
    b=load(p)
    scored=[]
    for r in b['outputs']:
        ok,pred,method=score_math_response(r['response'],b['solution'],20)
        scored.append({'correct':bool(ok),'predicted':pred,'method':method,'new_tokens':len(r['token_ids']),'finish_reason':r['finish_reason'],'think_tags':r['response'].count('<think>')+r['response'].count('</think>')})
    result={k:b[k] for k in ('dataset_index','problem_id','h','teacher_replicate','teacher_finished') if k in b}
    result.update(source_sha256=digest,outputs=scored,prefix_tokens=len(b.get('prefix_token_ids',[])))
    atomic(dest,result)
    return str(dest)

def stats(counts,n):
    return {'groups':len(counts),'samples_per_group':n,'correct':sum(counts),'rows':len(counts)*n,'accuracy':sum(counts)/(len(counts)*n),'avg_correct':sum(counts)/len(counts),'pass_any':sum(c>0 for c in counts)/len(counts),'pass_all':sum(c==n for c in counts)/len(counts),'correct_count_histogram':dict(sorted(Counter(counts).items()))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--limit',type=int,default=1000);ap.add_argument('--workers',type=int,default=32);args=ap.parse_args()
    out=args.output;cfg=load(out/'config.json')
    assert cfg['limit']==args.limit and cfg['scorer_sha256']==sha(ROOT/'opd_scripts/run_opd_logit_diagnostic.py')
    assert load(out/'STUDENT_GENERATION_COMPLETE.json')['rows']==args.limit*256
    rows=source(args.limit);tasks=[]
    for r in rows:
        i=int(r['dataset_index']);tp=teacher_path(out,i);tb=load(tp);validate_teacher(tb,r)
        tasks.append((str(tp),str(out/'scores/teacher'/tp.name)))
        for h in H:
            for t in tb['outputs']:
                p=student_path(out,h,i,t['teacher_replicate']);validate_student(load(p),r,t,h)
                tasks.append((str(p),str(out/f'scores/h{h}'/p.name)))
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        for k,_ in enumerate(pool.map(score_file,tasks,chunksize=1),1):
            if k%100==0:print('SCORED_BUNDLES',k,len(tasks),flush=True)
    teacher=[];all_scored=[];data={h:[] for h in H};per_question=[]
    for r in rows:
        i=int(r['dataset_index']);t=load(out/'scores/teacher'/f'{i:04d}.json');tc=sum(x['correct'] for x in t['outputs']);teacher.append(tc);all_scored+=t['outputs']
        q={'dataset_index':i,'problem_id':r['problem_id'],'teacher_correct':tc,'student_only_single_correct':bool(r['correct'])}
        for h in H:
            counts=[]
            for tr in range(8):
                b=load(out/f'scores/h{h}'/f'{i:04d}_{tr}.json');c=sum(x['correct'] for x in b['outputs']);counts.append(c);data[h].append(b);all_scored+=b['outputs']
            q[f'h{h}_per_teacher_correct']=counts
        per_question.append(q)
    summary={'teacher':stats(teacher,8),'student':{},'score_errors':sum(x['method'] in ('error','timeout') for x in all_scored),'length_stops':sum(x['finish_reason']=='length' for x in all_scored),'think_tags':sum(x['think_tags'] for x in all_scored),'scorer_script_sha256':sha(__file__),'student_only_single_accuracy':sum(bool(r['correct']) for r in rows)/len(rows)}
    for h in H:
        bs=data[h];counts=[sum(x['correct'] for x in b['outputs']) for b in bs]
        active=[sum(x['correct'] for x in b['outputs']) for b in bs if not b['teacher_finished']]
        perq=[sum(q[f'h{h}_per_teacher_correct']) for q in per_question]
        summary['student'][str(h)]={'conditional_on_teacher_prefix':stats(counts,8),'per_question_across_8_teacher_prefixes':stats(perq,64),'active_student_prefixes':stats(active,8) if active else None,'teacher_already_finished_prefixes':sum(b['teacher_finished'] for b in bs),'mean_full_tokens':sum(b['prefix_tokens']+x['new_tokens'] for b in bs for x in b['outputs'])/(len(bs)*8)}
    atomic(out/'per_question.json',per_question);atomic(out/'summary.json',summary)
    atomic(out/'SCORING_COMPLETE.json',{'scored_bundles':len(tasks),'questions':len(rows)})
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':
    import json
    main()
