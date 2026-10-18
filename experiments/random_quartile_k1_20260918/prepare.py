"""Freeze all r1_k1_t2 sources and uniformly sample within fixed quartiles."""
import collections, hashlib, json, random
from pathlib import Path
from ablation_core import atomic_json, digest
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo')
P=R/'opd_eval_ablation/random_quartile_k1_20260918'
SOURCES=[R/'opd_eval_code/code_grid_x8_20260916', R/'opd_eval_hard/math_reasoning_grid_x8_20260916']
RANDOM_SEED=2026091801

def plan(x, source, raw, row):
    n=len(x['response_token_ids']); ends=[n*i//4 for i in range(1,5)]; starts=[0]+ends[:-1]
    positions=[e['anchor_response_position'] for e in x['events'] if 'bridge_token_ids' in e]
    assert positions==sorted(set(positions)) and all(0<=a<n for a in positions)
    assert len(x['token_sources'])==n
    assert [i for i,t in enumerate(x['token_sources']) if t=='teacher']==positions
    counts=[sum(a<=v<b for v in positions) for a,b in zip(starts,ends)]
    rng=random.Random(digest([RANDOM_SEED,row['dataset'],row['problem_id'],row['dataset_index']]))
    selected=sorted(v for a,b,c in zip(starts,ends,counts) for v in rng.sample(range(a,b),c))
    assert len(selected)==len(positions)
    return dict(source=str(source),source_sha256=hashlib.sha256(raw).hexdigest(),source_finish_reason=x['finish_reason'],
                source_length=n,segment_end_token_counts=ends,intervention_counts=counts,
                original_positions_0based=positions,random_positions_0based=selected,random_seed=RANDOM_SEED)

def main():
    P.mkdir(parents=True,exist_ok=True)
    if (P/'INPUTS_COMPLETE.json').exists():
        m=json.loads((P/'INPUTS_COMPLETE.json').read_text());assert hashlib.sha256((P/'prompts.jsonl').read_bytes()).hexdigest()==m['sha256'];print(m);return
    rows=[]; manifests=[]
    for root in SOURCES:
        m=json.loads((root/'r1_k1_t2/manifest.json').read_text());manifests.append(m)
        assert m['settings']['gap_threshold']==2 and m['settings']['teacher_tokens']==1 and m['settings']['max_takeovers'] is None
        for row in map(json.loads,(root/'prompts.jsonl').read_text().splitlines()):
            f=root/'r1_k1_t2/completed'/f"{row['dataset_index']:05d}.json";raw=f.read_bytes();x=json.loads(raw)
            assert x['status']=='COMPLETE' and x['problem_id']==row['problem_id'] and x['prompt_token_ids']==row['prompt_token_ids']
            assert x['judge_calls']==0
            for e in x['events']:
                if 'bridge_token_ids' in e:assert len(e['bridge_token_ids'])==1 and e['selected_candidate']['signed_logprob_gap_student_minus_teacher']>2
            new=dict(row,quartile_plan=plan(x,f,raw,row),original_dataset_index=row['dataset_index'])
            new['dataset_index']=len(rows);rows.append(new)
    assert len(rows)==4571 and len({(r['dataset'],r['problem_id']) for r in rows})==4571
    settings=dict(manifests[0]['settings']);settings.update(mode='random_quartile',random_rates=[])
    assert all({k:v for k,v in m['settings'].items() if k!='random_rates'}=={k:v for k,v in manifests[0]['settings'].items() if k!='random_rates'} for m in manifests)
    raw=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode();(P/'prompts.jsonl').write_bytes(raw)
    atomic_json(P/'source_manifests.json',manifests);atomic_json(P/'base_settings.json',settings)
    atomic_json(P/'configs.json',[dict(name='r1_random_quartile_k1',k=1,threshold=2,seed=settings['seed'],replicate_index=0,base_config='random_quartile_k1')])
    meta=dict(rows=len(rows),sha256=hashlib.sha256(raw).hexdigest(),counts=dict(collections.Counter(r['dataset'] for r in rows)),source_finishes=dict(collections.Counter(r['quartile_plan']['source_finish_reason'] for r in rows)),random_seed=RANDOM_SEED,normal_eos=True,boundary_rule='floor(i*source_length/4), i=1..4; positions zero based, intervals [start,end)',interventions=sum(sum(r['quartile_plan']['intervention_counts']) for r in rows))
    atomic_json(P/'INPUTS_COMPLETE.json',meta);print(json.dumps(meta))
if __name__=='__main__':main()
