import json,sys,math,collections
from pathlib import Path

def quant(xs,p):
 if not xs:return None
 a=sorted(xs);i=(len(a)-1)*p;l=math.floor(i);h=math.ceil(i);return a[l]+(a[h]-a[l])*(i-l)
def stats(xs):
 return dict(n=len(xs),mean=sum(xs)/len(xs) if xs else None,p50=quant(xs,.5),p90=quant(xs,.9),p99=quant(xs,.99),max=max(xs) if xs else None)
def aggregate(paths,label,forced_ds=None):
 groups=collections.defaultdict(list)
 for p in paths:
  d=json.loads(p.read_text());groups[forced_ds or d.get('dataset','math500')].append(d)
 ans=[]
 for ds,rs in groups.items():
  counts=[];lengths=[];teachers=[];gaps=[];starts=[];first=[];offsets=[];relbins=[0]*10;tracebins=[0.]*10;offsetbins=[0]*4;rounds=0
  for d in rs:
   L=len(d['response_token_ids']);src=d['token_sources'];assert len(src)==L
   es=[e for e in d['events'] if e.get('bridge_token_ids')]
   ps=[e['anchor_response_position'] for e in es]
   assert ps==sorted(ps)
   counts.append(len(es));lengths.append(L);teachers.append(src.count('teacher'));rounds+=len(d['events'])
   hist=[0]*10
   for j,e in enumerate(es):
    pos=e['anchor_response_position'];k=len(e['bridge_token_ids']);assert src[pos:pos+k]==['teacher']*k
    b=min(9,int(pos/max(1,L)*10));relbins[b]+=1;hist[b]+=1;starts.append(pos)
    if 'response_offset' in e:
     off=pos-e['response_offset'];assert 0<=off<64;offsets.append(off);offsetbins[off//16]+=1
    if j:gaps.append(pos-(es[j-1]['anchor_response_position']+len(es[j-1]['bridge_token_ids'])))
   if es:
    first.append(ps[0]);tracebins=[a+b/len(es) for a,b in zip(tracebins,hist)]
  total=sum(counts);active=sum(c>0 for c in counts);T=sum(teachers);S=sum(lengths)-T
  out=dict(config=label,dataset=ds,questions=len(rs),takeovers=total,counts=stats(counts),zero=sum(c==0 for c in counts),count_bins={name:sum(lo<=c<=hi for c in counts) for name,lo,hi in [('0',0,0),('1-2',1,2),('3-5',3,5),('6-10',6,10),('11+',11,10**9)]},length=stats(lengths),teacher_tokens=T,teacher_tokens_mean=T/len(rs),teacher_share=T/sum(lengths),pooled_student_tokens_per_takeover=S/total if total else None,actual_between_bridge_student_gap=stats(gaps),student_segment_before_takeover=stats(gaps+first),absolute_position=stats(starts),first_position=stats(first),relative_position_decile_counts=relbins,relative_position_decile_percent=[100*x/total if total else 0 for x in relbins],trace_weighted_decile_percent=[100*x/active if active else 0 for x in tracebins],chunk_offset=stats(offsets),chunk_offset_quartile_counts=offsetbins,rounds=rounds,intervened_round_share=total/rounds)
  ans.append(out)
 return ans
if __name__=='__main__':
 if '--remote' in sys.argv:
  r=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath');out=[]
  for k,name in [(8,'online_chunk64_k8_test1000_2nodes_seed20260911'),(32,'online_chunk64_k32_test1000_h100_seed20260911')]:
   paths=list((r/name/'results/completed').glob('*.json'));assert len(paths)==1000
   out+=aggregate(paths,f'Luna_abs2_k{k}','deepmath')
  print(json.dumps(out))
 else:
  root=Path(__file__).resolve().parents[2];out=[]
  for directory,label,expected in [('hard_three_20260911','Luna_abs2_k8',1814),('hard_k32_20260911','Luna_abs2_k32',1814),('threshold_k8_nojudge_20260911','NoLuna_abs2_k8',2314),('threshold3_mathteacher_20260911','NoLuna_abs3_k8',2314)]:
   paths=list((root/'experiments'/directory/'artifacts/online/completed').glob('*.json'));assert len(paths)==expected
   out+=aggregate(paths,label)
  paths=list((root/'experiments/math500_three_20260911/results/online/completed').glob('*.json'));assert len(paths)==500
  out+=aggregate(paths,'Luna_abs2_k8','math500')
  Path(__file__).with_name('local_summary.json').write_text(json.dumps(out,indent=2))
  print('Aggregated',sum(x['questions'] for x in out),'traces across',len(out),'config/dataset groups')
