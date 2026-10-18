import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from generate import BASE, read, atomic
from run_opd_logit_diagnostic import score_math_response
OUT=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/teacher_prefix_handoff_20260909')

def score(row):
    if 'correct' not in row:
        ok,pred,method=score_math_response(row['response'],row['solution'],20)
        row.update(correct=bool(ok),predicted_answer=pred,score_method=method)
    return row

def main():
    teachers=sorted(read(BASE/'qwen3_14b/merged/rollouts.jsonl'),key=lambda r:int(r['dataset_index']))
    students=sorted(read(BASE/'qwen3_1_7b/merged/rollouts.jsonl'),key=lambda r:int(r['dataset_index']))
    assert len(teachers)==len(students)==1000
    results=[]
    with ProcessPoolExecutor(max_workers=16,mp_context=mp.get_context('spawn')) as pool:
        for h in (32,64,128,256):
            path=OUT/f'h{h}.unscored.json' if h==32 else OUT/f'parallel/h{h}/h{h}.unscored.json'
            rows=json.loads(path.read_text())
            assert [int(r['dataset_index']) for r in rows]==list(range(1000))
            for r,t in zip(rows,teachers):
                assert r['problem_id']==t['problem_id'] and r['solution']==t['solution']
                assert r['prefix_token_ids']==t['response_token_ids'][:h]
            rows=list(pool.map(score,rows,chunksize=1))
            atomic(OUT/f'h{h}.json',rows)
            correct=sum(r['correct'] for r in rows)
            result={'h':h,'n':1000,'correct':correct,'accuracy':correct/1000,'delta_pp_vs_student':(correct-283)/10,'rescued':sum(r['correct'] and not s['correct'] for r,s in zip(rows,students)),'harmed':sum(not r['correct'] and s['correct'] for r,s in zip(rows,students)),'teacher_already_finished':sum(r['teacher_finished_before_handoff'] for r in rows),'length_stops':sum(r['finish_reason']=='length' for r in rows),'score_timeouts_or_errors':sum(r['score_method'] in ('timeout','error') for r in rows)}
            results.append(result)
            atomic(OUT/'summary.json',results)
            print(result,flush=True)
    (OUT/'COMPLETE').write_text('verified four conditions, 1000 each\n')

if __name__=='__main__':main()
