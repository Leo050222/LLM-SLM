import json,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
BASE=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath')
CODE=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_online_code/20260911_chunk64_k8_2nodes/ocrd_pipeline')
sys.path.insert(0,str(CODE))
from full_rollout.analyze_teacher_rollout_shard import score_math_response
from online_rollout.core import atomic_json,digest
OUT=BASE/'takeover_baseline_comparison'
def score(row):
    key=digest([row['response'],row['solution'],20.,(CODE/'full_rollout/analyze_teacher_rollout_shard.py').read_text()])
    p=OUT/'baseline_scores'/f"{row['dataset_index']:05d}.json"
    if p.exists():
        v=json.loads(p.read_text())
        if v['identity']==key:return v
    correct,_,method=score_math_response(row['response'],row['solution'],20.)
    v=dict(dataset_index=row['dataset_index'],correct=correct,method=method,identity=key,original_correct=row['correct'])
    atomic_json(p,v);return v

def main():
    rows=[json.loads(l) for l in (BASE/'ge6_test1000_native_nonthinking_seed20260811/qwen3_1_7b/merged/rollouts.jsonl').open() if l.strip()]
    assert len(rows)==1000 and {r['dataset_index'] for r in rows}==set(range(1000))
    original={r['dataset_index']:r for r in rows}
    from verl.utils.reward_score.math_reward import strip_string
    strip_string('1')
    with ThreadPoolExecutor(max_workers=16) as pool:scored=list(pool.map(score,rows))
    baseline={r['dataset_index']:r for r in scored}
    result=dict(baseline_correct=sum(r['correct'] for r in scored),baseline_original_correct=sum(r['original_correct'] for r in scored),rescoring_changes=sum(r['correct']!=r['original_correct'] for r in scored),experiments={})
    for k,name in [(8,'online_chunk64_k8_test1000_2nodes_seed20260911'),(32,'online_chunk64_k32_test1000_h100_seed20260911')]:
        root=BASE/name
        prompts=[json.loads(l) for l in (root/'prompts1000.jsonl').read_text().splitlines()]
        for p in prompts:
            b=original[p['dataset_index']]
            assert str(p['problem_id'])==str(b['problem_id']) and p['problem']==b['problem'] and str(p['solution'])==str(b['solution']) and p['prompt_token_ids']==b['prompt_token_ids']
        scores=[json.loads(p.read_text()) for p in (root/'results/scores').glob('*.json')]
        assert len(scores)==1000
        bins=[]
        for lo,hi in [(0,0),(1,1),(2,2),(3,4),(5,7),(8,12),(13,100000),(1,100000),(0,100000)]:
            v=[r for r in scores if lo<=r['takeovers']<=hi];n=len(v)
            bc=sum(baseline[r['dataset_index']]['correct'] for r in v);ac=sum(r['correct'] for r in v)
            gain=sum(not baseline[r['dataset_index']]['correct'] and r['correct'] for r in v)
            loss=sum(baseline[r['dataset_index']]['correct'] and not r['correct'] for r in v)
            bins.append(dict(range=[lo,hi],n=n,baseline_correct=bc,assisted_correct=ac,baseline_accuracy=bc/n,assisted_accuracy=ac/n,wrong_to_right=gain,right_to_wrong=loss))
        result['experiments'][str(k)]=bins
    atomic_json(OUT/'summary.json',result)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
