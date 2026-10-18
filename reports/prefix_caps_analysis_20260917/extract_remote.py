import json,csv,hashlib,collections,time
from pathlib import Path
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_prefix');roots=[R/'prefix_takeover_caps_20260917',R/'prefix_takeover_caps5_10_20260917'];out=R/'caps0_10_analysis_20260917';out.mkdir(exist_ok=True)
raw=[(p/'prompts.jsonl').read_bytes() for p in roots];assert raw[0]==raw[1]
meta=[json.loads((p/'INPUTS_COMPLETE.json').read_text()) for p in roots]
assert meta[0]['branch_seeds']==meta[1]['branch_seeds']
rows={r['dataset_index']:r for r in map(json.loads,raw[0].splitlines())};seen=set();agg=collections.defaultdict(collections.Counter);statuses=collections.Counter();n=0
with (out/'paired_scores.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['dataset','sample_index','problem_id','cap','replicate','correct','actual_takeovers','status']);w.writeheader()
 for p,m in zip(roots,meta):
  assert (p/'ALL_GENERATION_COMPLETE.json').exists() and (p/'ALL_SCORING_COMPLETE.json').exists()
  for cap in m['caps']:
   for rep in range(8):
    d=p/f'cap{cap}_r{rep}';files=list((d/'scores').glob('*.json'));assert len(files)==800
    for file in files:
     x=json.loads(file.read_text());i=x['sample_index'];row=rows[i]
     assert type(x['correct']) is bool and x['cap']==cap and x['replicate']==rep
     assert x['dataset']==row['dataset'] and x['problem_id']==row['problem_id']
     assert 0<=x['actual_takeovers']<=cap and (d/'completed'/file.name).exists()
     key=(i,cap,rep);assert key not in seen;seen.add(key)
     status=x['status'] if isinstance(x['status'],str) else json.dumps(x['status'],sort_keys=True)
     w.writerow(dict(dataset=x['dataset'],sample_index=i,problem_id=x['problem_id'],cap=cap,replicate=rep,correct=int(x['correct']),actual_takeovers=x['actual_takeovers'],status=status))
     g=agg[(x['dataset'],cap)];g['n']+=1;g['correct']+=x['correct'];g['actual_takeovers_total']+=x['actual_takeovers'];statuses[status]+=1;n+=1
  for g in json.loads((p/'summary.json').read_text())['groups']:
   assert dict(agg[(g['dataset'],g['cap'])])=={k:g[k] for k in ['n','correct','actual_takeovers_total']}
  print('EXTRACTED',p.name,n,flush=True)
assert n==70400 and len(seen)==70400
manifest=[json.loads((p/'manifest.json').read_text()) for p in roots];assert manifest[0]['models']==manifest[1]['models']
audit={'rows':n,'unique_prefixes':len(rows),'input_sha256':hashlib.sha256(raw[0]).hexdigest(),'paired_scores_sha256':hashlib.sha256((out/'paired_scores.csv').read_bytes()).hexdigest(),'source_summaries_reconciled':True,'same_model_identities':True,'same_prefixes_and_seeds':True,'status_counts':dict(statuses),'extracted_at':time.time(),'verification_scope':'All score rows and summary counts checked; original output hashes are recorded by scorer but not rehashed here'}
(out/'extraction_audit.json').write_text(json.dumps(audit));print(json.dumps(audit),flush=True)
