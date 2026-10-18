import json,csv
from pathlib import Path
from decimal import Decimal,ROUND_HALF_UP
import numpy as np
P=Path(__file__).parent;old=json.loads((P.parent/'prefix_caps0_20_analysis_20260918/pass8_analysis.json').read_text())['metrics'];new=json.loads((P/'remote_evidence.json').read_text());caps=[25,30,35,40];ds=list(old['accuracy']['datasets']);names=['BBH','CRUXEval-I','CRUXEval-O','HumanEval+','LiveCodeBench','MATH500','MBPP+','OlympiadBench'];ids={d:sorted({r[1] for r in new['rows'] if r[0]==d}) for d in ds};assert all(len(v)==100 for v in ids.values())
A=np.full((8,100,4,8),np.nan)
for d,i,c,r,y,t in new['rows']:
 ix=(ds.index(d),ids[d].index(i),caps.index(c),r);assert np.isnan(A[ix]);A[ix]=y
assert np.isfinite(A).all() and len(new['rows'])==25600
M={'accuracy':A.mean(axis=(1,3))*100,'pass_at_8':A.any(axis=3).mean(axis=1)*100,'pass_all_8':A.all(axis=3).mean(axis=1)*100}
headers=['数据集','Student-only（0）']+[str(i) for i in range(1,21)]+list(map(str,caps))+['Teacher-only'];result={};lines=['单位：%。Student-only（0）为固定student prefix后续写；Teacher-only为从原题生成。每档每数据集100题×8次。新增25/30/35/40档逐条核验25600评分及汇总、样本、种子与模型身份；未重新评分原始输出。各批次GPU部署不同；新增worker支持KV缓存大小配置并启用统计日志，worker哈希因此变化，模型清单其余字段一致。编号配对不意味着逐token嵌套轨迹。',''];short=[]
def f(x):return str(Decimal(str(round(float(x),6))).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))
for key,title in [('accuracy','平均 acc'),('pass_at_8','pass@8'),('pass_all_8','pass all 8')]:
 data=[old[key]['datasets'][d][:21]+M[key][i].tolist()+[old[key]['datasets'][d][21]] for i,d in enumerate(ds)];data.append(np.mean(data,axis=0).tolist());result[key]={'columns':[0]+list(range(1,21))+caps+['teacher'],'datasets':dict(zip(ds,data[:8])),'overall':data[8]}
 with (P/f'{key}.csv').open('w') as file:
  w=csv.writer(file);w.writerow(headers);w.writerows([[name]+[f(x) for x in v] for name,v in zip(names+['总体'],data)])
 lines+=['**'+title+'**','','| '+' | '.join(headers)+' |','|---|'+'---:|'*(len(headers)-1)]
 short+=['**'+title+'**','','| 数据集 | Student-only（0） | 20 | 25 | 30 | 35 | 40 | Teacher-only |','|---|'+'---:|'*7]
 for name,v in zip(names+['总体'],data):
  lines.append('| '+name+' | '+' | '.join(f(x) for x in v)+' |');short.append('| '+name+' | '+' | '.join(f(v[c]) for c in [0,20,21,22,23,24,25])+' |')
 lines+=[''];short+=['']
(P/'FULL_TABLES_WITH_BASELINES.md').write_text('\n'.join(lines)+'\n');(P/'NEW_CAPS_COMPARISON.md').write_text('\n'.join(short)+'\n');(P/'metrics.json').write_text(json.dumps(result,indent=2));print('\n'.join(short))
