from pathlib import Path
import json,csv,itertools,os
from collections import Counter
import numpy as np
P=Path(__file__).resolve().parent;R=P.parents[1]
baselines=json.loads((P/'baselines-paired-source.json').read_text())
B={(r['dataset'],r['config'],r['problem_id']):r['correct'] for r in baselines}
assert len(B)==len(baselines)
math=json.loads((R/'reports/grid_math_rescore_comparison_20260916/paired_results.json').read_text())['paired']
math=[r for r in math if r['config']=='k1_t2']
code=json.loads((R/'experiments/dynamic_k_entropy_20260914/results/extension_vs_k1/paired.json').read_text())
code=[r for r in code if r['dataset'] not in ['math500','olympiad_bench','bbh']]
H={(r['dataset'],r['problem_id']):r['new_correct'] for r in math}
H.update({(r['dataset'],r['problem_id']):r['k1_correct'] for r in code})
assert len(H)==4571
order=['math500','olympiad_bench','bbh','cruxeval_o','cruxeval_i','humaneval_plus','mbpp_plus','livecodebench']
names=['MATH500','OlympiadBench','BBH','CRUXEval-O','CRUXEval-I','HumanEval+','MBPP+','LiveCodeBench']
expected=[500,674,1080,800,800,164,378,175]
patterns=list(itertools.product([True,False],repeat=3));counts={};paired=[]
raw=json.loads((R/'outputs/benchmark-viz/data.json').read_text());reference={r['name']:r for r in raw['rows']}
for j,ds in enumerate(order):
 ids={pid for d,pid in H if d==ds}
 for g in ['student_only','teacher_only']:assert ids=={pid for d,c,pid in B if d==ds and c==g},(ds,g)
 assert len(ids)==expected[j]
 c=Counter()
 for pid in sorted(ids):
  pattern=(B[ds,'student_only',pid],H[ds,pid],B[ds,'teacher_only',pid]);assert all(type(x)is bool for x in pattern);c[pattern]+=1
  paired.append(dict(dataset=ds,problem_id=pid,student_correct=pattern[0],intervention_correct=pattern[1],teacher_correct=pattern[2]))
 counts[ds]=c
 for idx,g in enumerate(['student_only','k1_t2','teacher_only']):
  ncorrect=sum(n for pat,n in c.items() if pat[idx]);assert abs(100*ncorrect/len(ids)-reference[g]['v'][j][0])<.011,(ds,g,ncorrect)
 assert sum(c.values())==expected[j]
headers=['Student-only','Intervention\nk=1, t=2','Teacher-only']+[f'{b}\nN={n}' for b,n in zip(names,expected)]
body=[];long=[]
for pattern in patterns:
 row=['Correct' if b else 'Wrong' for b in pattern]
 for ds,n in zip(order,expected):
  count=counts[ds][pattern];row.append(f'{count} ({100*count/n:.2f}%)');long.append(dict(dataset=ds,student_correct=pattern[0],intervention_correct=pattern[1],teacher_correct=pattern[2],count=count,total=n,percent=100*count/n))
 body.append(row)
for name,rr in [('k1-t2-combinations.csv',long),('k1-t2-paired-samples.csv',paired)]:
 with (P/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
md=['# 三种模式的逐题正确性组合（k=1，阈值=2）','','每格：数量（占本数据集全部样本的比例）。列顺序：Student-only → Teacher介入 → Teacher-only。','', '|Student-only|Teacher介入|Teacher-only|'+'|'.join(f'{b} (n={n})' for b,n in zip(names,expected))+'|','|'+'---|'*11]
for pat,row in zip(patterns,body):md.append('|'+'|'.join(['对' if x else '错' for x in pat]+row[3:])+'|')
md+=['','数学与BBH三种模式均采用2026-09-16 GPT-4o-mini重评分；代码采用原测试评分。未剔除超时样本。按(dataset, problem_id)严格一一匹配，4571题无缺失、无重复，三种模式边际准确率均与当前总表一致。','Teacher介入表示该配置完整运行后的最终结果，包括实际零介入样本；非仅统计发生过介入的子集。不同停止规则/评分误差仍可能影响结果，组合计数不代表因果效应。']
(P/'k1-t2-combinations.md').write_text('\n'.join(md))
os.environ['MPLCONFIGDIR']='/private/tmp/benchmark-viz-mpl'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(23,5.7));ax.axis('off')
table=ax.table(cellText=body,colLabels=headers,cellLoc='center',colWidths=[.085,.095,.085]+[.091875]*8,bbox=[0,.15,1,.76])
table.auto_set_font_size(False);table.set_fontsize(10)
for (r,c),cell in table.get_celld().items():
 cell.set_edgecolor('#dbe3e9');cell.set_linewidth(.5)
 if r==0:cell.set_facecolor('#e8eff5');cell.set_text_props(weight='bold',color='#183247')
 elif c<3:
  cell.set_facecolor('#e4f1e9' if patterns[r-1][c] else '#f7e7e6');cell.set_text_props(color='#183247')
 else:cell.set_facecolor('#f5f8fa' if r%2 else '#ffffff');cell.set_text_props(color='#183247')
fig.suptitle('Per-question correctness combinations | Student → Intervention → Teacher',x=.04,y=.97,ha='left',fontsize=20)
fig.text(.04,.88,'Intervention: k=1, threshold=2. Each cell: count (% of all questions in that benchmark).',fontsize=12,color='#475569')
fig.text(.04,.045,'Exact matching by dataset + problem ID; 4,571 questions, no missing or duplicate matches. All marginal accuracies match the current summary.\nMath / BBH: GPT-4o-mini rescoring (2026-09-16); code: original test scores. Includes zero-intervention questions and timeouts under existing scoring rules.',fontsize=10,color='#475569')
fig.subplots_adjust(left=.04,right=.99,top=.89,bottom=.11)
fig.savefig(P/'k1-t2-combinations.png',dpi=200,bbox_inches='tight');fig.savefig(P/'k1-t2-combinations.pdf',bbox_inches='tight');plt.close(fig)
print('\n'.join(md))
