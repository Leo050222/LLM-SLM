import os,json
from pathlib import Path
os.environ['MPLCONFIGDIR']='/tmp/opd_matplotlib'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
P=Path(__file__).parent
O=P/'dataset_lineplots';O.mkdir(exist_ok=True)
data=json.loads((P/'pass8_analysis.json').read_text())['metrics']
names={'bbh':'BBH','cruxeval_i':'CRUXEval-I','cruxeval_o':'CRUXEval-O','humaneval_plus':'HumanEval+','livecodebench':'LiveCodeBench','math500':'MATH500','mbpp_plus':'MBPP+' ,'olympiad_bench':'OlympiadBench'}
metrics=[('accuracy','Mean accuracy','#2563a6'),('pass_at_8','Pass@8','#16806a'),('pass_all_8','Pass all 8','#8653ad')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
def panel(ax,ds,key,title,color):
 vals=data[key]['datasets'][ds];assert len(vals)==22
 ax.plot(range(21),vals[:21],color=color,marker='o',markersize=4,linewidth=2,label='Student + teacher interventions')
 ax.axhline(vals[21],color='#d47928',linestyle='--',linewidth=1.7,label=f'Teacher: {vals[21]:.2f}%')
 ax.set_title(title,fontweight='bold',pad=12);ax.set_xlabel('Maximum teacher interventions');ax.set_ylabel('Percent (%)');ax.set_xticks([0,2,4,6,8,10,12,14,16,18,20]);ax.set_xlim(-.6,20.6)
 ax.set_ylim(0,100);ax.set_yticks(range(0,101,20));ax.grid(alpha=.18);ax.legend(loc='best',fontsize=8,framealpha=.9)
 ax.annotate(f'{vals[20]:.2f}',(20,vals[20]),xytext=(-4,8),textcoords='offset points',ha='right',fontsize=9,color=color,fontweight='bold')
with PdfPages(O/'all_datasets_three_metrics.pdf') as pdf:
 for ds,label in names.items():
  fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
  for ax,(key,title,color) in zip(axes,metrics):panel(ax,ds,key,title,color)
  fig.suptitle(label,fontsize=20,fontweight='bold')
  fig.supxlabel('100 problems × 8 rollouts per budget. Teacher starts from the original prompt; intervention runs share a frozen student prefix. All panels use the same 0–100% y-axis.',fontsize=9,color='#536273')
  fig.savefig(O/f'{ds}_three_metrics.png',dpi=180);fig.savefig(O/f'{ds}_three_metrics.pdf');pdf.savefig(fig);plt.close(fig)
  for key,title,color in metrics:
   fig,ax=plt.subplots(figsize=(6.4,4.6),layout='constrained');panel(ax,ds,key,f'{label} · {title}',color);fig.savefig(O/f'{ds}_{key}.png',dpi=180);plt.close(fig)
(O/'README.md').write_text('# Per-dataset line plots\n\nEight datasets, three panels each: mean accuracy, empirical pass@8 (any of 8 correct), pass all 8 (all 8 correct). X-axis: intervention cap 0–20. Dashed line: teacher-only from the original prompt. Intervention runs share a frozen student prefix. Each dataset has 100 problems × 8 rollouts/cap. All panels use the same 0–100% y-axis; all values are percentages.\n\nSource: ../pass8_analysis.json. Combined PDF: all_datasets_three_metrics.pdf (8 pages). Each dataset also has a three-panel PNG/PDF and three standalone PNG files.\n')
print('Created 8 three-panel PNG/PDFs, 24 standalone PNGs, and an 8-page PDF.')
