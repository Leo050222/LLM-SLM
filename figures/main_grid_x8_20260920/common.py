from pathlib import Path
import csv, json, re, hashlib
from collections import defaultdict, Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
DS=dict(zip(['math500','olympiad_bench','bbh','cruxeval_o','cruxeval_i','humaneval_plus','mbpp_plus','livecodebench'],['MATH500','OlympiadBench','BBH','CRUXEval-O','CRUXEval-I','HumanEval+','MBPP+','LiveCodeBench']))
KS=[1,4,8,16,32]; TS=[2,2.5,3,3.5,4]
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#7B6F00']; MARKERS=['o','s','^','D','P']
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':10,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
SOURCES=['reports/math_grid_x8_results_20260919/all_metrics.csv','reports/code_grid_x8_results_20260918/pass8_results.json','reports/code_grid_x8_results_20260918/generation_totals.json','reports/full_baseline_x8_status_20260918/pass8_results.json','reports/full_baseline_x8_status_20260918/code_tokens.json']
def load(p):return json.loads((ROOT/p).read_text())
def data():
 rows={}
 for r in csv.DictReader((ROOT/SOURCES[0]).open()):
  rows[r['config'],r['dataset']]={k:(v if k in ['config','dataset'] else float(v)) for k,v in r.items()}
 totals=defaultdict(Counter)
 for g in load(SOURCES[2])['groups']:
  for d,v in g['datasets'].items():totals[g['config'],d].update(v)
 for g in load(SOURCES[4]):
  for d,v in g['totals'].items():totals[g['role']+'_only',d].update(v)
 for r in load(SOURCES[1])['results']+load(SOURCES[3])['results']:
  c,d=r['config'],r['dataset']
  if d not in DS or d in list(DS)[:3]:continue
  a=totals[c,d];n=r['questions'];h=r['correct_count_histogram']
  assert a['n']==8*n and sum(h)==n
  assert np.isclose(sum(i*v for i,v in enumerate(h))/(8*n),r['mean_accuracy'])
  rows[c,d]=dict(config=c,dataset=d,questions=n,rollouts=8*n,accuracy_pct=100*r['mean_accuracy'],pass_at_8_pct=100*r['pass_at_8'],pass_all_8_pct=100*r['pass_all_8'],mean_interventions=a['interventions']/a['n'],teacher_token_pct=100*a['teacher_tokens']/a['output_tokens'],mean_output_tokens=a['output_tokens']/a['n'])
 expected={'student_only','teacher_only'}|{f'k{k}_t{t:g}' for t in TS for k in KS}
 for d in DS:
  assert {c for c,dd in rows if dd==d}==expected
  assert len({r['questions'] for (c,dd),r in rows.items() if dd==d})==1
  for c in expected:
   r=rows[c,d];assert r['rollouts']==8*r['questions']
   assert 0<=r['pass_all_8_pct']<=r['accuracy_pct']<=r['pass_at_8_pct']<=100
   assert 0<=r['teacher_token_pct']<=100
 assert len(rows)==216
 return rows
DATA=data()
def grid(d):return [DATA[f'k{k}_t{t:g}',d] for t in TS for k in KS]
def panel(ax,d,i):
 ax.text(0,1.08,f'({chr(97+i)}) {DS[d]}',transform=ax.transAxes,weight='bold',fontsize=11)
 ax.text(1,1.08,f'n={int(DATA["student_only",d]["questions"])} × 8',ha='right',transform=ax.transAxes,fontsize=8,color='#555555')
def save(fig,name):
 for ext in ['pdf','png','svg']:fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight',pad_inches=.12)
 plt.close(fig)
def export():
 checks=0
 for line in (ROOT/'reports/math_grid_x8_results_20260919/all_benchmarks_six_metrics.md').read_text().splitlines()[2:]:
  cells=[v.strip() for v in line.strip('|').split('|')]
  for d,cell in zip(DS,cells[3:]):
   for key,val in zip(['accuracy_pct','pass_at_8_pct','pass_all_8_pct','mean_interventions','teacher_token_pct','mean_output_tokens'],cell.split('/')):
    assert abs(DATA[cells[0],d][key]-float(val.strip().rstrip('%')))<=.00500001
    checks+=1
 keys=['dataset','config','questions','rollouts','accuracy_pct','pass_at_8_pct','pass_all_8_pct','mean_interventions','teacher_token_pct','mean_output_tokens']
 with (OUT/'plot_data.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(DATA.values())
 (OUT/'validation.json').write_text(json.dumps({'original_main_table_crosscheck':{'values':checks,'mismatches':0,'tolerance':.00500001},'rows':len(DATA),'datasets':8,'grid_configurations':25,'baselines':'complete x8, no historical single-run baseline','checks':['complete 27 × 8 coverage','8 rollouts per question','pass_all_8 <= accuracy <= pass_at_8','code histogram matches accuracy','code token counts match 8N'],'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}},indent=2))
if __name__=='__main__':export()
