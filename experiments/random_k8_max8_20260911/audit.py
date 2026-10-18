from pathlib import Path
import json,hashlib,collections,random
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');P=R/'opd_eval_hard/random_k8_max8_20260911'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert (P/'SCORING_COMPLETE').exists();rows=[json.loads(l) for l in (P/'prompts.jsonl').open()];assert len(rows)==len({r['problem_id'] for r in rows})==2314
assert sha(P/'prompts.jsonl')==load(P/'manifest.json')['prompt_sha256']==load(P/'data_manifest.json')['prompts_sha256']
assert collections.Counter(r['dataset'] for r in rows)==dict(math500=500,aime24=30,aime25=30,olympiad_bench=674,bbh=1080)
assert len(list((P/'online/completed').glob('*.json')))==len(list((P/'scores/online').glob('*.json')))==2314
assert load(P/'ALL_GENERATION_COMPLETE.json')['rows']==2314;gate=(P/'ALL_GENERATION_COMPLETE.json').stat().st_mtime
stats={};correct=collections.Counter()
for r in rows:
 i=r['dataset_index'];base=R/('opd_eval_math500/math500_three_20260911' if r['dataset']=='math500' else 'opd_eval_hard/hard_three_20260911');bf=base/'student/completed'/f"{r['source_dataset_index']:05d}.json";bv=load(bf);assert sha(bf)==r['reference_student_sha256'] and bv['prompt_token_ids']==r['prompt_token_ids']
 n=len(bv['response_token_ids']);assert n==r['reference_student_length'];k=min(8,(n+7)//8);rng=random.Random(f'20260911:random_anchors:{i}');anchors=[x+7*j for j,x in enumerate(sorted(rng.sample(range(n-7*(k-1)),k)))] if k else [];assert anchors==r['random_anchor_offsets']
 f=P/'online/completed'/f'{i:05d}.json';v=load(f);sf=P/'scores/online'/f.name;s=load(sf)
 assert v['status']=='COMPLETE' and v['problem_id']==r['problem_id']==s['problem_id'] and v['prompt_token_ids']==r['prompt_token_ids'] and v['solution']==r['solution']
 assert s['source_sha256']==sha(f) and f.stat().st_mtime<=gate<=sf.stat().st_mtime and s['status']=='ok'
 ev=[e for e in v['events'] if 'bridge_token_ids' in e];assert len(ev)==v['teacher_count']<=8;assert [e['anchor_response_position'] for e in ev]==anchors[:len(ev)]
 assert v['judge_calls']==0 and len(v['token_sources'])==len(v['response_token_ids'])
 for e in ev:
  t=e['bridge_token_ids'];pos=e['anchor_response_position'];assert 0<len(t)<=8 and t[0]!=e['student_entry_token'] and v['response_token_ids'][pos:pos+len(t)]==t and v['token_sources'][pos:pos+len(t)]==['teacher']*len(t)
 ds=r['dataset'];a=stats.setdefault(ds,dict(n=0,correct=0,tokens=0,teacher_tokens=0,takeovers=0,max_takeovers=0,zero_takeovers=0,finish_reasons={}))
 a['n']+=1;a['correct']+=bool(s['correct']);a['tokens']+=len(v['response_token_ids']);a['teacher_tokens']+=v['token_sources'].count('teacher');a['takeovers']+=len(ev);a['max_takeovers']=max(a['max_takeovers'],len(ev));a['zero_takeovers']+=len(ev)==0;a['finish_reasons'][v['finish_reason']]=a['finish_reasons'].get(v['finish_reason'],0)+1
summary=load(P/'summary.json')['online']
for ds,a in stats.items():assert summary[ds]['correct']==a['correct'] and summary[ds]['questions']==a['n']
print(json.dumps(dict(verified=True,rows=2314,statistics=stats),indent=2))
