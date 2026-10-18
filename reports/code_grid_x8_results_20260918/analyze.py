import json,csv,statistics,math
from pathlib import Path
P=Path(__file__).resolve().parent
s=json.loads((P/'x8_score_summary.json').read_text()); marker=json.loads((P/'ALL_X8_SCORING_COMPLETE.json').read_text())
assert s['rows']==marker['rows']==463400 and s['scoring_signature']==marker['signature']
ds={'cruxeval_o':('CRUXEval-O',800),'cruxeval_i':('CRUXEval-I',800),'humaneval_plus':('HumanEval+',164),'mbpp_plus':('MBPP+',378),'livecodebench':('LiveCodeBench',175)}
combined={(r['dataset'],r['config']):r for r in s['combined']}
assert len(combined)==125 and len(s['groups'])==1000 and sum(r['n'] for r in s['groups'])==463400
for (d,c),r in combined.items():
 g=[x for x in s['groups'] if x['dataset']==d and x['config']==c]
 assert len(g)==8 and {x['replicate'] for x in g}==set(range(8)) and all(x['n']==ds[d][1] for x in g)
 assert sum(x['correct'] for x in g)==r['correct'] and r['n']==ds[d][1]*8
 assert math.isclose(statistics.mean(x['accuracy'] for x in g),r['accuracy']) and math.isclose(statistics.stdev(x['accuracy'] for x in g),r['replicate_accuracy_sd'])
b=json.loads((P.parent.parent/'outputs/benchmark-viz/data.json').read_text())
base={r['name']:{d:r['v'][b['bench'].index(label)][0] for d,(label,n) in ds.items()} for r in b['rows'] if r['name'] in ['student_only','teacher_only']}
configs=sorted({c for d,c in combined},key=lambda c:(float(c.split('_t')[1]),int(c.split('_')[0][1:])))
lines=['# Code 排列组合：8 次 rollout 评分汇总','', '已核验 25 个配置 × 5 个数据集 × 8 次，共 463,400 条评分；1000 个 replicate 分组齐全。完成标记与汇总评分签名一致。','', '每格为准确率均值 ± 8 次 rollout 的样本标准差（%，标准差单位为百分点）。这是所有生成的平均正确率，不是 pass@8；重复采样针对相同题目，标准差不表示独立题目抽样的置信区间。生成超时按原评分规则计错。Student/Teacher 为历史单次基线，不属于本轮 x8。','', '| 配置 | '+' | '.join(x[0] for x in ds.values())+' |','|---|'+'---:|'*5]
for name in ['student_only','teacher_only']:
 lines.append('| '+name+'（历史单次） | '+' | '.join(f'{base[name][d]:.2f}' for d in ds)+' |')
for c in configs:
 lines.append('| '+c+' | '+' | '.join(f"{combined[d,c]['accuracy']*100:.2f} ± {combined[d,c]['replicate_accuracy_sd']*100:.2f}" for d in ds)+' |')
lines+=['','## 各数据集最优组合','', '| 数据集 | 最优组合 | 正确数 / 总数 | 均值 ± SD | 相对 Student（百分点 / 相对提升） | 相对 Teacher（百分点） |','|---|---|---:|---:|---:|---:|']
for d,(label,n) in ds.items():
 best=max(combined[d,c]['accuracy'] for c in configs); winners=[c for c in configs if combined[d,c]['accuracy']==best];r=combined[d,winners[0]]
 delta=best*100-base['student_only'][d]
 lines.append(f"| {label} | {', '.join(winners)} | {r['correct']}/{r['n']} | {best*100:.2f} ± {' / '.join(format(combined[d,c]['replicate_accuracy_sd']*100,'.2f') for c in winners)} | +{delta:.2f} / +{delta/base['student_only'][d]*100:.1f}% | {best*100-base['teacher_only'][d]:+.2f} |")
lines+=['','单次基线的差异仅作描述性对照，不能据此判断显著性；最优配置是同一网格内选出的观测最大值。不同数据集不合并成一个总准确率。','', '来源：远端 opd_eval_code/code_grid_x8_20260916/x8_score_summary.json、ALL_X8_SCORING_COMPLETE.json；历史基线 outputs/benchmark-viz/data.json。','', '后续如需严谨比较 Teacher，建议在相同完整题集及相同生成条件下补齐基线 8 次 rollout，再做按题配对比较。']
(P/'REPORT.md').write_text('\n'.join(lines)+'\n')
with (P/'x8_score_summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(s['combined'][0]));w.writeheader();w.writerows(s['combined'])
print('\n'.join(lines))
