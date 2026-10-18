from pathlib import Path
import json,statistics,collections
r=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_code/code_reasoning_grid_20260913');data={};output={'stats':{},'comparisons':{},'examples':[]}
for c in ['student_only','teacher_only','k1_t2']:
 for p in (r/c/'completed').glob('*.json'):
  x=json.loads(p.read_text());ds=x['dataset']
  if ds not in ['cruxeval_i','cruxeval_o']:continue
  score=json.loads((r/c/'scores'/p.name).read_text());x['_correct']=score['correct'];data.setdefault((c,ds),{})[x['problem_id']]=x
 for ds in ['cruxeval_i','cruxeval_o']:
  rs=list(data[c,ds].values());ns=sorted(len(x['response_token_ids'])for x in rs);normal=[len(x['response_token_ids'])for x in rs if x['finish_reason']=='stop'];tag='INPUT' if ds.endswith('_i') else 'OUTPUT'
  first=[x['events'][0].get('anchor_response_position') for x in rs if x.get('events')];first=[v for v in first if v is not None]
  output['stats'][c+'/'+ds]={'n':len(ns),'mean':statistics.mean(ns),'median':statistics.median(ns),'p90':ns[int(.9*(len(ns)-1))],'p95':ns[int(.95*(len(ns)-1))],'p99':ns[int(.99*(len(ns)-1))],'max':max(ns),'gt100':sum(n>100 for n in ns),'gt1000':sum(n>1000 for n in ns),'top10_share':sum(ns[-10:])/sum(ns),'finish':dict(collections.Counter(x['finish_reason']for x in rs)),'normal_mean':statistics.mean(normal),'any_answer_tag':sum('['+tag+']' in x['response']for x in rs),'repeated_answer_tags':sum(x['response'].count('['+tag+']')>1 for x in rs),'mean_correct_length':statistics.mean([len(x['response_token_ids'])for x in rs if x['_correct']]),'mean_wrong_length':statistics.mean([len(x['response_token_ids'])for x in rs if not x['_correct']]),'first_intervention_median':statistics.median(first)if first else None}
for ds in ['cruxeval_i','cruxeval_o']:
 a,b=data['student_only',ds],data['k1_t2',ds];assert a.keys()==b.keys()
 ids=sorted(a,key=lambda key:len(a[key]['response_token_ids']),reverse=True)
 total=sum(len(a[k]['response_token_ids'])-len(b[k]['response_token_ids'])for k in ids)
 bothstop=[k for k in ids if a[k]['finish_reason']==b[k]['finish_reason']=='stop']
 short=[k for k in ids if max(len(a[k]['response_token_ids']),len(b[k]['response_token_ids']))<=1000]
 output['comparisons'][ds]={'reduction_tokens':total,'top10_baseline_share_of_reduction':sum(len(a[k]['response_token_ids'])-len(b[k]['response_token_ids'])for k in ids[:10])/total,'both_stop_n':len(bothstop),'both_stop_means':[statistics.mean(len(d[k]['response_token_ids'])for k in bothstop)for d in [a,b]],'both_le1000_n':len(short),'both_le1000_means':[statistics.mean(len(d[k]['response_token_ids'])for k in short)for d in [a,b]],'shorter_equal_longer':[sum(len(b[k]['response_token_ids'])<len(a[k]['response_token_ids'])for k in ids),sum(len(b[k]['response_token_ids'])==len(a[k]['response_token_ids'])for k in ids),sum(len(b[k]['response_token_ids'])>len(a[k]['response_token_ids'])for k in ids)]}
 samples=ids[:3]+[k for k in ids if 100<=len(a[k]['response_token_ids'])<=1000][:3]+[k for k in ids if 20<=len(a[k]['response_token_ids'])<100][:2]
 for k in samples:
  v={'dataset':ds,'id':k,'problem':a[k]['problem']}
  for name,d in [('student',a),('k1',b),('teacher',data['teacher_only',ds])]:
   x=d[k];v[name]={'n':len(x['response_token_ids']),'finish':x['finish_reason'],'correct':x['_correct'],'head':x['response'][:1100],'tail':x['response'][-500:] if len(x['response'])>1100 else '', 'first_event':x.get('events',[])[:1] if name=='k1' else []}
  output['examples'].append(v)
print(json.dumps(output))
