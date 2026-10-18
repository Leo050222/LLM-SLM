from pathlib import Path
import json,math,numpy as np,csv
P=Path(__file__).parent;rs=json.loads((P/'positions.json').read_text());assert len(rs)==4571
assert all(r['length']==r['source_length'] for r in rs)
print('event_position_mismatch',sum(r['positions']!=r['event_positions']for r in rs))
def expected(n,m,d):
 if n-d<m:return 1.
 return -math.expm1(sum(math.log1p(-d/(n-i))for i in range(m)))
def summarize(a):
 gaps=[];exp={d:0 for d in [1,4,8,16]}; thirds=np.zeros(3);tthird=[];span=[];individual=[]
 for r in a:
  pos=np.array(r['positions']);m=len(pos);n=r['length']
  if not m:continue
  bins=np.minimum(2,(3*pos/max(n,1)).astype(int));counts=np.bincount(bins,minlength=3);thirds+=counts;tthird.append(counts/m)
  if m<2:continue
  gs=np.diff(pos);gaps.extend(gs.tolist());span.append((pos[-1]-pos[0])/max(n-1,1))
  individual.append(float(np.mean(gs<=8)))
  for d in exp:exp[d]+=(m-1)*expected(n,m,d)
 out={'n':len(a),'intervened':sum(bool(r['positions'])for r in a),'multi':len(span),'events':int(sum(thirds)),'gaps':len(gaps),'median_gap':float(np.median(gaps)),'gap_p90':float(np.quantile(gaps,.9)),'median_span_fraction':float(np.median(span)),'thirds_event_weighted':(thirds/sum(thirds)).tolist(),'thirds_trace_weighted':np.mean(tthird,axis=0).tolist(),'trace_mean_gap_le8':float(np.mean(individual))}
 for d in exp:out['gap_le'+str(d)]=sum(g<=d for g in gaps)/len(gaps);out['uniform_expected_le'+str(d)]=exp[d]/len(gaps)
 return out
out={ds:summarize([r for r in rs if r['dataset']==ds])for ds in sorted(set(r['dataset']for r in rs))};out['ALL']=summarize(rs);out['NORMAL_ONLY']=summarize([r for r in rs if r['finish']=='stop'])
(P/'summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
with (P/'summary.csv').open('w')as f:
 w=csv.DictWriter(f,fieldnames=['dataset']+list(out['ALL']));w.writeheader();w.writerows(dict(dataset=ds,**v)for ds,v in out.items())
