"""Run on downloaded artifacts only; never deploy into the hashed runtime directory."""
import csv,hashlib,json,sys,statistics
from pathlib import Path
from collections import Counter
P=Path(sys.argv[1]);configs=json.loads((P/'configs.json').read_text());rows=[json.loads(l) for l in (P/'prompts.jsonl').read_text().splitlines()]
assert len(configs)==25 and {(c['k'],c['threshold']) for c in configs}=={(k,t) for k in [1,4,8,16,32] for t in [2,2.5,3,3.5,4]}
assert Counter(r['dataset'] for r in rows)==dict(math500=500,olympiad_bench=674,bbh=1080)
assert len({r['dataset_index'] for r in rows})==2254
gate=P/'ALL_GENERATION_COMPLETE.json';assert json.loads(gate.read_text())['rows']==56350
report=[]
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
for c in configs:
 name=c['name'];p=P/name;m=json.loads((p/'manifest.json').read_text());s=m['settings']
 assert s['teacher_tokens']==c['k'] and s['gap_threshold']==c['threshold'] and s['max_takeovers'] is None
 assert s['mode']=='nojudge_grid' and s['gap_rule']=='student_minus_teacher_gt_positive_threshold' and s['first_token_ban']
 assert m['judge_config'] is None and m['thinking'] is False and m['sampling']==dict(temperature=1.,top_p=1.)
 assert m['expected_rows']==2254 and m['input_sha256']==digest(rows)
 assert s['seed']==20260911 and s['chunk_tokens']==64 and s['max_tokens']==40264
 assert len(list((p/'completed').glob('*.json')))==2254 and len(list((P/'scores'/name).glob('*.json')))==2254
 data=[]
 for row in rows:
  f=p/'completed'/f"{row['dataset_index']:05d}.json";r=json.loads(f.read_text());sf=P/'scores'/name/f.name;sc=json.loads(sf.read_text())
  assert sc['source_sha256']==hashlib.sha256(f.read_bytes()).hexdigest()
  assert r['status']=='COMPLETE' and r['judge_calls']==0 and r['identity']==digest(dict(row=row,settings=s))
  for k,v in row.items():assert r[k]==v,(name,f.name,k)
  assert sc['dataset_index']==row['dataset_index'] and sc['problem_id']==row['problem_id'] and sc['dataset']==row['dataset']
  # User authorized scoring completed groups before the global generation gate.
  # Validate source hashes rather than the superseded all-generation-first rule.
  if r.get('finish_reason')=='trace_timeout':
   assert r['timeout_count_as_wrong'] is True and r['trace_elapsed_seconds']>=900
   assert sc['correct'] is False and sc['status']=='trace_timeout'
   assert m['runtime_policy']['trace_limit_seconds']==900
  else:
   assert sc['status']!='trace_timeout'
  ids=r['response_token_ids'];sources=r['token_sources'];assert len(ids)==len(sources)<=40264
  events=[e for e in r['events'] if 'bridge_token_ids' in e]
  assert all(not e['judgments'] for e in r['events'])
  for e in events:
   b=e['bridge_token_ids'];candidate=e['selected_candidate'];pos=e['anchor_response_position']
   assert 0<len(b)<=c['k'] and b[0]!=e['student_entry_token']
   assert candidate['signed_logprob_gap_student_minus_teacher']>c['threshold']
   assert ids[pos:pos+len(b)]==b and sources[pos:pos+len(b)]==['teacher']*len(b)
  assert sources.count('teacher')==sum(len(e['bridge_token_ids']) for e in events)
  data.append(dict(dataset=row['dataset'],correct=sc['correct'],score_status=sc['status'],tokens=len(ids),teacher_tokens=sources.count('teacher'),takeovers=len(events),finish=r.get('finish_reason')))
 for ds in ['math500','olympiad_bench','bbh']:
  rr=[r for r in data if r['dataset']==ds];n=len(rr);tok=sum(r['tokens'] for r in rr);tt=sum(r['teacher_tokens'] for r in rr)
  report.append(dict(config=name,k=c['k'],threshold=c['threshold'],dataset=ds,n=n,correct=sum(r['correct'] for r in rr),accuracy=sum(r['correct'] for r in rr)/n,mean_tokens=tok/n,median_tokens=statistics.median(r['tokens'] for r in rr),mean_takeovers=sum(r['takeovers'] for r in rr)/n,teacher_token_share=tt/tok,score_status=dict(Counter(r['score_status'] for r in rr)),finish_counts=dict(Counter(r['finish'] for r in rr))))
(P/'audit_summary.json').write_text(json.dumps(dict(status='PASS',configs=25,outputs=56350,scores=56350,stopping_policy='Mixed: prior completed traces retained; later unfinished traces have900second active wall-time deadline, timeout counted incorrect',scoring_policy='User authorized completed-group early scoring; source hashes verified',results=report),indent=2))
with (P/'comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(report[0]));w.writeheader();w.writerows(report)
print('PASS: all25 configurations,56350 outputs/scores and protocol/source/timeout-policy checks')
