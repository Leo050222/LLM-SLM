from pathlib import Path
import json,hashlib,collections,random
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');P=R/'opd_eval_hard/threshold_k8_nojudge_20260911'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert (P/'SCORING_COMPLETE').exists();rows=[json.loads(l) for l in (P/'prompts.jsonl').open()];assert len(rows)==len({r['problem_id'] for r in rows})==2314
assert load(P/'manifest.json')['judge'] is False
assert collections.Counter(r['dataset'] for r in rows)==dict(math500=500,aime24=30,aime25=30,olympiad_bench=674,bbh=1080)
assert len(list((P/'online/completed').glob('*.json')))==len(list((P/'scores/online').glob('*.json')))==2314
assert load(P/'ALL_GENERATION_COMPLETE.json')['rows']==2314;gate=(P/'ALL_GENERATION_COMPLETE.json').stat().st_mtime
stats={};correct=collections.Counter()
for r in rows:
 i=r['dataset_index'];base=R/('opd_eval_math500/math500_three_20260911' if r['dataset']=='math500' else 'opd_eval_hard/hard_three_20260911');bf=base/'student/completed'/f"{r['source_dataset_index']:05d}.json";bv=load(bf);assert bv['prompt_token_ids']==r['prompt_token_ids'] and bv['solution']==r['solution']
 f=P/'online/completed'/f'{i:05d}.json';v=load(f);sf=P/'scores/online'/f.name;s=load(sf)
 assert v['status']=='COMPLETE' and v['problem_id']==r['problem_id']==s['problem_id'] and v['prompt_token_ids']==r['prompt_token_ids'] and v['solution']==r['solution']
 assert s['source_sha256']==sha(f) and f.stat().st_mtime<=gate<=sf.stat().st_mtime and s['status']=='ok'
 ev=[e for e in v['events'] if 'bridge_token_ids' in e];assert all(not e['judgments'] for e in v['events'])
 assert v['judge_calls']==0 and len(v['token_sources'])==len(v['response_token_ids'])
 for e in ev:
  t=e['bridge_token_ids'];pos=e['anchor_response_position'];assert 0<len(t)<=8 and t[0]!=e['threshold_candidate']['student_token_id'] and v['response_token_ids'][pos:pos+len(t)]==t and v['token_sources'][pos:pos+len(t)]==['teacher']*len(t)
  c=e['threshold_candidate'];assert c['candidate_index']==0 and c['absolute_logprob_gap']>2 and abs(c['student_logprob']-c['teacher_logprob_on_student_token'])>2
 ds=r['dataset'];a=stats.setdefault(ds,dict(n=0,correct=0,tokens=0,teacher_tokens=0,takeovers=0,max_takeovers=0,zero_takeovers=0,finish_reasons={}))
 a['n']+=1;a['correct']+=bool(s['correct']);a['tokens']+=len(v['response_token_ids']);a['teacher_tokens']+=v['token_sources'].count('teacher');a['takeovers']+=len(ev);a['max_takeovers']=max(a['max_takeovers'],len(ev));a['zero_takeovers']+=len(ev)==0;a['finish_reasons'][v['finish_reason']]=a['finish_reasons'].get(v['finish_reason'],0)+1
summary=load(P/'summary.json')['online']
for ds,a in stats.items():assert summary[ds]['correct']==a['correct'] and summary[ds]['questions']==a['n']
print(json.dumps(dict(verified=True,rows=2314,statistics=stats),indent=2))
