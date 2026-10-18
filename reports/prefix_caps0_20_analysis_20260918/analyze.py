import os,csv,json
from pathlib import Path
os.environ['MPLCONFIGDIR']='/tmp/opd_matplotlib'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).parent
new=json.loads((P/'remote_evidence.json').read_text())
old=list(csv.DictReader((P.parent/'prefix_caps_analysis_20260917/paired_scores.csv').open()))
rows=[{k:(v if k=='dataset' else int(v)) for k,v in r.items() if k in new['columns']} for r in old]+[dict(zip(new['columns'],r)) for r in new['rows']]
names=sorted(set(r['dataset'] for r in rows));ids={d:sorted(set(r['sample_index'] for r in rows if r['dataset']==d)) for d in names}
A=np.full((8,100,21,8),np.nan);T=A.copy()
for r in rows:
 ix=(names.index(r['dataset']),ids[r['dataset']].index(r['sample_index']),r['cap'],r['replicate']);assert np.isnan(A[ix]);A[ix]=r['correct'];T[ix]=r['actual_takeovers']
assert len(rows)==134400 and np.isfinite(A).all() and np.isfinite(T).all()
teach=json.loads((P.parent/'teacher_vs_prefix_caps_20260918/summary.json').read_text());tv=np.array([next(g['accuracy'] for g in teach['groups'] if g['dataset']==d)*100 for d in names]);acc=A.mean(axis=(1,3))*100;overall=acc.mean(axis=0)
rng=np.random.default_rng(20260918);boots=[]
for d in range(8):
 w=rng.multinomial(100,np.full(100,.01),size=10000)/100;boots.append(w@A[d].mean(axis=2)*100)
boots=np.array(boots);allboot=boots.mean(axis=0)
result={'overall_accuracy':overall.tolist(),'overall_mean_actual':T.mean(axis=(0,1,3)).tolist(),'teacher':float(tv.mean()),'datasets':{}}
for d,name in enumerate(names):
 result['datasets'][name]={'accuracy':acc[d].tolist(),'teacher':float(tv[d]),'gain10_20':float(acc[d,20]-acc[d,10]),'ci10_20':np.quantile(boots[d,:,20]-boots[d,:,10],[.025,.975]).tolist(),'actual20':float(T[d,:,20].mean()),'cap20_exhausted_pct':float((T[d,:,20]==20).mean()*100),'gap_closed_pct':float((acc[d,20]-acc[d,0])/(tv[d]-acc[d,0])*100)}
for a,b in [(0,10),(10,15),(15,20),(10,20),(0,20)]:result[f'gain{a}_{b}']={'pp':float(overall[b]-overall[a]),'ci':np.quantile(allboot[:,b]-allboot[:,a],[.025,.975]).tolist()}
result['overall_gap_closed_pct']=float((overall[20]-overall[0])/(tv.mean()-overall[0])*100)
(P/'analysis.json').write_text(json.dumps(result,indent=2))
labels=['BBH','CRUXEval-I','CRUXEval-O','HumanEval+','LiveCodeBench','MATH500','MBPP+','OlympiadBench']
with (P/'comparison.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['dataset']+list(range(21))+['teacher']);w.writerows([[n]+list(v)+[t] for n,v,t in zip(labels,acc,tv)]);w.writerow(['Overall']+list(overall)+[tv.mean()])
lines=['# 0–20次介入预算实验分析','','134400条评分完整覆盖800题×8次×21档。新增64000条逐条核对唯一键、布尔评分、题目身份、介入次数范围，并与汇总计数一致。远程输入、种子、模型身份与原0–4档一致。未重新评分或重新哈希rollout正文。','','Teacher从原始题目生成，介入组从固定student prefix生成。正确率为逐次rollout指标，非pass@8。95%区间按各数据集100题整簇重采样10000次，保留8次重复和所有预算配对；以现有种子为条件，未校正多重比较。跨预算编号配对不保证逐token嵌套轨迹。','','| Dataset | 0 | 10 | 15 | 20 | Teacher | 10→20(pp) | 95% CI |','|---|---:|---:|---:|---:|---:|---:|---|']
for d,n in enumerate(labels):
 r=result['datasets'][names[d]];lines.append('| '+n+' | '+' | '.join(f'{x:.2f}' for x in [acc[d,0],acc[d,10],acc[d,15],acc[d,20],tv[d],r['gain10_20']])+f" | [{r['ci10_20'][0]:.2f}, {r['ci10_20'][1]:.2f}] |")
lines+=['','总体准确率（cap0至cap20）：'+', '.join(f'{v:.2f}' for v in overall),'','总体差值与置信区间：',json.dumps({k:v for k,v in result.items() if k.startswith('gain')},ensure_ascii=False,indent=2),'','完整表：comparison.csv。实际介入次数、预算耗尽比例和分数据集统计：analysis.json。','', '![curves](accuracy_curves.png)']
(P/'REPORT.md').write_text('\n'.join(lines)+'\n')
fig,axs=plt.subplots(3,3,figsize=(13,10),layout='constrained');caps=np.arange(21)
for d,ax in enumerate(axs.flat):
 y=overall if d==8 else acc[d];t=tv.mean() if d==8 else tv[d];ax.plot(caps,y,'o-',ms=3,lw=1.7);ax.axhline(t,color='#dd7733',ls='--',label='Teacher from start');ax.axvline(10,color='gray',lw=.8,alpha=.5);ax.set_title('Overall' if d==8 else labels[d]);ax.set_xticks([0,5,10,15,20]);ax.grid(alpha=.2);ax.set_xlabel('Intervention budget');ax.set_ylabel('Accuracy (%)');ax.legend(fontsize=8)
fig.savefig(P/'accuracy_curves.png',dpi=160);plt.close(fig)
print(json.dumps(result,indent=2))
