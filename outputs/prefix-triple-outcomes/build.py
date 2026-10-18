from pathlib import Path
from collections import Counter
import json,csv,itertools,os
os.environ['MPLCONFIGDIR']='/private/tmp/benchmark-viz-mpl'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
P=Path(__file__).resolve().parent;R=P.parents[1]
source=R/'reports/prefix_caps_analysis_20260917/paired_scores.csv'
rows=list(csv.DictReader(source.open()))
lookup={(r['dataset'],r['problem_id'],int(r['cap']),int(r['replicate'])):r['correct']=='1' for r in rows}
assert len(lookup)==len(rows)==70400
B=json.loads((R/'outputs/triple-outcomes/baselines-paired-source.json').read_text())
T={(r['dataset'],r['problem_id']):r['correct'] for r in B if r['config']=='teacher_only'}
order=['math500','olympiad_bench','bbh','cruxeval_o','cruxeval_i','humaneval_plus','mbpp_plus','livecodebench'];names=['MATH500','OlympiadBench','BBH','CRUXEval-O','CRUXEval-I','HumanEval+','MBPP+','LiveCodeBench']
patterns=list(itertools.product([True,False],repeat=3));long=[];paired=[]
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
with PdfPages(P/'prefix-combinations-caps1-10.pdf') as pdf:
 for cap in range(1,11):
  counts={}
  for ds in order:
   keys=sorted((pid,rep) for d,pid,c,rep in lookup if d==ds and c==cap)
   assert len(keys)==800 and len({p for p,r in keys})==100
   counter=Counter()
   for pid,rep in keys:
    pat=(lookup[ds,pid,0,rep],lookup[ds,pid,cap,rep],T[ds,pid]);counter[pat]+=1
    paired.append(dict(dataset=ds,problem_id=pid,replicate=rep,cap=cap,student_cap0_correct=pat[0],intervention_correct=pat[1],historical_teacher_correct=pat[2]))
   assert sum(counter.values())==800
   assert sum(n for pat,n in counter.items() if pat[2])==8*sum(T[ds,pid] for pid in {p for p,r in keys})
   counts[ds]=counter
  body=[]
  for pat in patterns:
   line=['Correct' if v else 'Wrong' for v in pat]
   for ds in order:
    n=counts[ds][pat];line.append(f'{n} ({100*n/800:.2f}%)');long.append(dict(cap=cap,dataset=ds,student_cap0_correct=pat[0],intervention_correct=pat[1],historical_teacher_correct=pat[2],count=n,total_rollouts=800,unique_prefixes=100,percent=100*n/800))
   body.append(line)
  headers=['Student continuation\ncap=0',f'Intervention\ncap={cap}','Teacher-only reference\nHistorical, whole question']+[b+'\n100 prefixes × 8' for b in names]
  fig,ax=plt.subplots(figsize=(24,6));ax.axis('off')
  table=ax.table(cellText=body,colLabels=headers,cellLoc='center',colWidths=[.09,.09,.125]+[.086875]*8,bbox=[0,.17,1,.73]);table.auto_set_font_size(False);table.set_fontsize(10)
  for (i,j),cell in table.get_celld().items():
   cell.set_edgecolor('#dbe3e9');cell.set_linewidth(.5)
   if i==0:cell.set_facecolor('#e8eff5');cell.set_text_props(weight='bold',color='#183247')
   elif j<3:cell.set_facecolor('#e4f1e9' if patterns[i-1][j] else '#f7e7e6');cell.set_text_props(color='#183247')
   else:cell.set_facecolor('#f5f8fa' if i%2 else '#ffffff');cell.set_text_props(color='#183247')
  fig.suptitle(f'Fixed-prefix rollout combinations | Intervention cap = {cap}',x=.035,y=.97,ha='left',fontsize=21)
  fig.text(.035,.88,'k=1, threshold=2 | Count (% of 800 paired continuations per dataset); 100 prefixes, 8 replicate seeds.',fontsize=12,color='#475569')
  fig.text(.035,.045,'Student cap=0 and intervention are matched by prefix + replicate index. Teacher reference is one historical whole-question result per problem, reused across 8 replicates.\nThis is NOT a three-way same-prefix rollout comparison, NOT pass@8, and NOT 800 independent problems. Paired seeds do not guarantee identical token trajectories.',fontsize=10,color='#475569')
  fig.subplots_adjust(left=.035,right=.99,top=.89,bottom=.11);pdf.savefig(fig);fig.savefig(P/f'prefix-combinations-cap{cap}.pdf',bbox_inches='tight');fig.savefig(P/f'prefix-combinations-cap{cap}.png',dpi=180,bbox_inches='tight');plt.close(fig)
  md=[f'# 固定 prefix 三列组合：介入上限 {cap}', '', '列顺序：Student cap=0续写 → Teacher介入续写 → 历史整题Teacher-only参照。每格为次数（占800次续写的比例），不是100题级pass@8。','', '|Student cap=0|Teacher介入|历史Teacher-only|'+'|'.join(names)+'|','|'+'---|'*11]
  md+=['|'+'|'.join(['对' if v else '错' for v in pat]+line[3:])+'|' for pat,line in zip(patterns,body)]
  (P/f'cap{cap}.md').write_text('\n'.join(md))
for name,records in [('combinations-all-caps.csv',long),('paired-continuations.csv',paired)]:
 with (P/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
(P/'README.md').write_text('''# 固定 prefix 续写与历史 Teacher-only 的三列组合

每数据集100题，每题固定prefix续写8次。k=1、阈值=2，分别统计已核验完整的介入上限1–10；Student列为同prefix、同replicate编号的cap=0续写。

Teacher-only列来自原始整题生成的同题历史结果，每题只有一个布尔标签，重复用于8个replicate作为参照。不是同prefix下Teacher续写8次，不能解读为严格三模式反事实对照。数学与BBH使用GPT-4o-mini重评分口径；代码用测试评分。

每列分母800，100个独立prefix簇；百分比是逐次rollout组合占比，不是题目级成功率或pass@8。同种子不保证逐token轨迹一致；仅作编号配对描述。样本限于存在首次触发点的条件化队列。

源数据：reports/prefix_caps_analysis_20260917/paired_scores.csv；outputs/triple-outcomes/baselines-paired-source.json。

已检查70400条记录唯一键、各档cap=0完整配对、每数据集100prefix/800rollout、Teacher历史标签完整覆盖及重复8次的边际计数。
''')
print('Created 10 tables / 640 combination cells / 64000 matched continuation triples.')
