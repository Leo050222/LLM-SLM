import os,json,csv
from pathlib import Path
os.environ['MPLCONFIGDIR']='/tmp/opd_matplotlib'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).parent
new=json.loads((P/'remote_evidence.json').read_text());old=list(csv.DictReader((P.parent/'prefix_caps_analysis_20260917/paired_scores.csv').open()))
rows=[{k:(v if k=='dataset' else int(v)) for k,v in r.items() if k in new['columns']} for r in old]+[dict(zip(new['columns'],r)) for r in new['rows']]
names=sorted({r['dataset'] for r in rows});labels=['BBH','CRUXEval-I','CRUXEval-O','HumanEval+','LiveCodeBench','MATH500','MBPP+','OlympiadBench'];ids={d:sorted({r['sample_index'] for r in rows if r['dataset']==d}) for d in names}
A=np.full((8,100,22,8),np.nan)
for r in rows:
 ix=(names.index(r['dataset']),ids[r['dataset']].index(r['sample_index']),r['cap'],r['replicate']);assert np.isnan(A[ix]);A[ix]=r['correct']
prompts={r['dataset_index']:r for r in map(json.loads,(P.parent/'prefix_caps_analysis_20260917/prompts.jsonl').read_text().splitlines())}
for r in json.loads((P/'teacher_paired_scores.json').read_text()):
 assert r['problem_id']==prompts[r['sample_index']]['problem_id'] and r['dataset']==prompts[r['sample_index']]['dataset']
 ix=(names.index(r['dataset']),ids[r['dataset']].index(r['sample_index']),21,r['replicate']);assert np.isnan(A[ix]);A[ix]=r['correct']
assert np.isfinite(A).all() and set(np.unique(A))=={0,1}
C=A.sum(axis=3);metrics={'accuracy':C/8,'pass_at_8':(C>0).astype(float),'pass_all_8':(C==8).astype(float)};result={'definition':'Empirical any-correct / all-correct among 8 rollouts per problem; each dataset 100 problems; overall 800 problems. Teacher starts from original prompt; caps use frozen student prefix.','bootstrap':'10000 stratified paired problem-cluster resamples; conditional on existing seeds, no multiple-comparison correction.','metrics':{}}
rng=np.random.default_rng(20260918);weights=[rng.multinomial(100,np.full(100,.01),size=10000)/100 for _ in names]
for name,M in metrics.items():
 ds=M.mean(axis=1)*100;overall=ds.mean(axis=0);boot=np.array([w@m*100 for w,m in zip(weights,M)]);B=boot.mean(axis=0);res={'datasets':{d:ds[i].tolist() for i,d in enumerate(names)},'overall':overall.tolist(),'changes':{}}
 for a,b in [(0,10),(10,20),(0,20),(15,20),(21,20)]:
  diff=M[:,:,b]-M[:,:,a];res['changes'][f'{a}_to_{b}']={'pp':float(overall[b]-overall[a]),'ci':np.quantile(B[:,b]-B[:,a],[.025,.975]).tolist(),'gained_problems':int((diff>0).sum()),'lost_problems':int((diff<0).sum()),'datasets':{d:{'pp':float(ds[i,b]-ds[i,a]),'ci':np.quantile(boot[i,:,b]-boot[i,:,a],[.025,.975]).tolist()} for i,d in enumerate(names)}}
 result['metrics'][name]=res
 with (P/f'{name}_0_20_teacher.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['dataset']+list(range(21))+['Teacher']);w.writerows([[l]+list(v) for l,v in zip(labels,ds)]);w.writerow(['Overall']+list(overall))
result['correct_count_histograms']={str(c):np.bincount(C[:,:,c].astype(int).ravel(),minlength=9).tolist() for c in [0,10,15,20,21]}
(P/'pass8_analysis.json').write_text(json.dumps(result,indent=2))
fig,axs=plt.subplots(1,3,figsize=(15,4.3),layout='constrained')
for ax,(key,title) in zip(axs,[('accuracy','Mean accuracy'),('pass_at_8','Pass@8 (at least one correct)'),('pass_all_8','All 8 correct')]):
 y=result['metrics'][key]['overall'];ax.plot(range(21),y[:21],'o-',color='#2459a6',ms=4);ax.axhline(y[21],color='#d97732',ls='--',label=f'Teacher: {y[21]:.2f}%');ax.axvline(10,color='gray',alpha=.35);ax.set_title(title);ax.set_xlabel('Maximum interventions');ax.set_ylabel('Percent');ax.set_xticks([0,5,10,15,20]);ax.grid(alpha=.2);ax.legend()
fig.savefig(P/'pass8_overall_curves.png',dpi=170);plt.close(fig)
lines=['# Pass@8与八次全对率：0–20次介入','','每题8次rollout中至少1次正确定义为pass@8，全部8次正确定义为pass all 8。每个数据集分母100题，总体800题；直接由逐题评分计算，不由平均acc推算。Teacher从原题生成，cap组从固定student prefix生成。','','| Cap | Mean accuracy | Pass@8 | All 8 correct |','|---|---:|---:|---:|']
for c in list(range(21))+[21]:lines.append('| '+('Teacher' if c==21 else str(c))+' | '+' | '.join(f'{result["metrics"][k]["overall"][c]:.2f}' for k in metrics)+' |')
for key in ['pass_at_8','pass_all_8']:
 lines+=['',key,'','| Dataset | 0 | 10 | 15 | 20 | Teacher |','|---|---:|---:|---:|---:|---:|']
 for d,l in zip(names,labels):lines.append('| '+l+' | '+' | '.join(f'{result["metrics"][key]["datasets"][d][c]:.2f}' for c in [0,10,15,20,21])+' |')
lines+=['','置信区间：10000次按数据集分层、按题目配对整簇bootstrap；保留8次rollout，以现有种子为条件，未做多重比较校正。跨预算配对不保证逐token嵌套反事实。八次全对是有限样本稳定性指标，不代表对该题永远正确。','', '![Overall curves](pass8_overall_curves.png)','', '完整统计及正确次数0–8分布：pass8_analysis.json；全21档表：pass_at_8_0_20_teacher.csv、pass_all_8_0_20_teacher.csv。']
(P/'PASS8_REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:38]))
for key in ['pass_at_8','pass_all_8']:
 print(key)
 for d,l in zip(names,labels):print(l,[round(result['metrics'][key]['datasets'][d][c],3) for c in [0,10,15,20,21]])
 for pair in ['0_to_10','10_to_20','0_to_20','15_to_20','21_to_20']:
  r=result['metrics'][key]['changes'][pair];print(pair,{k:v for k,v in r.items() if k!='datasets'})
print('HIST',result['correct_count_histograms'])
