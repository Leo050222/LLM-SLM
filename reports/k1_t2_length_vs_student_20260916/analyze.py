from pathlib import Path
import json,csv,statistics,collections
P=Path(__file__).parent
x=json.loads((P/'paired_evidence.json').read_text());rows=x['rows']
order=['math500','olympiad_bench','bbh','cruxeval_o','cruxeval_i','humaneval_plus','mbpp_plus','livecodebench']

def clean(r):
 return all(r[m]['finish']=='stop' and 'timeout' not in str(r[m]['score_status']).lower() for m in ['student','k1'])

def big(r):return r['k1']['tokens']<=r['student']['tokens']*.5 and r['student']['tokens']-r['k1']['tokens']>=100

def acc(rs,m):return sum(r[m]['correct']for r in rs)/len(rs) if rs else None

def quant(ns,p):
 a=sorted(ns);return a[int(p*(len(a)-1))]if a else None

summary=[]
for ds in order:
 rs=[r for r in rows if r['dataset']==ds];n=len(rs);selected=[r for r in rs if big(r)];actual=[r for r in selected if r['interventions']>0];normal=[r for r in rs if clean(r)];normbig=[r for r in normal if big(r)];a=[r['student']['tokens']for r in rs];b=[r['k1']['tokens']for r in rs];delta=sum(a)-sum(b)
 top=sorted(rs,key=lambda r:r['student']['tokens'],reverse=True)[:10]
 rec=dict(dataset=ds,n=n,actually_intervened=sum(r['interventions']>0 for r in rs),student_mean=statistics.mean(a),k1_mean=statistics.mean(b),mean_reduction_fraction=1-sum(b)/sum(a),student_median=statistics.median(a),k1_median=statistics.median(b),student_p90=quant(a,.9),k1_p90=quant(b,.9),student_p99=quant(a,.99),k1_p99=quant(b,.99),shorter=sum(y<z for z,y in zip(a,b)),halved=sum(y<=z*.5 for z,y in zip(a,b)),big_short_n=len(selected),big_short_intervened_n=len(actual),big_short_fraction=len(selected)/n,big_short_student_acc=acc(selected,'student'),big_short_k1_acc=acc(selected,'k1'),wrong_to_right=sum(not r['student']['correct'] and r['k1']['correct']for r in selected),right_to_wrong=sum(r['student']['correct'] and not r['k1']['correct']for r in selected),big_short_mean_student=statistics.mean(r['student']['tokens']for r in selected)if selected else None,big_short_mean_k1=statistics.mean(r['k1']['tokens']for r in selected)if selected else None,big_short_token_reduction_contribution=sum(r['student']['tokens']-r['k1']['tokens']for r in selected)/delta if delta else None,big_short_intervention_median=statistics.median(r['interventions']for r in actual)if actual else None,both_stop_no_timeout_n=len(normal),both_stop_student_mean=statistics.mean(r['student']['tokens']for r in normal),both_stop_k1_mean=statistics.mean(r['k1']['tokens']for r in normal),both_stop_big_short_n=len(normbig),both_stop_big_student_acc=acc(normbig,'student'),both_stop_big_k1_acc=acc(normbig,'k1'),student_gt1000=sum(v>1000 for v in a),k1_gt1000=sum(v>1000 for v in b),student_top10_share=sum(r['student']['tokens']for r in top)/sum(a),top10_net_reduction_contribution=sum(r['student']['tokens']-r['k1']['tokens']for r in top)/delta if delta else None,student_finish=dict(collections.Counter(r['student']['finish']for r in rs)),k1_finish=dict(collections.Counter(r['k1']['finish']for r in rs)))
 rec.update(big_short_intervened_fraction=len(actual)/n,big_short_intervened_student_acc=acc(actual,'student'),big_short_intervened_k1_acc=acc(actual,'k1'),big_short_intervened_wrong_to_right=sum(not r['student']['correct'] and r['k1']['correct']for r in actual),big_short_intervened_right_to_wrong=sum(r['student']['correct'] and not r['k1']['correct']for r in actual))
 summary.append(rec)
(P/'summary.json').write_text(json.dumps(summary,indent=2))
with (P/'summary.csv').open('w',newline='')as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
flat=[]
for r in rows:
 rec={k:v for k,v in r.items()if k not in ['student','k1']}
 for m in ['student','k1']:
  for k,v in r[m].items():rec[m+'_'+k]=json.dumps(v)if isinstance(v,dict)else v
 rec.update(big_short=big(r),both_stop_no_timeout=clean(r));flat.append(rec)
with (P/'paired.csv').open('w',newline='')as f:
 w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
for r in summary:print(json.dumps(r,ensure_ascii=False))
