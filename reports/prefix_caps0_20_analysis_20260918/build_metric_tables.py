import os,json,csv
from pathlib import Path
import numpy as np
os.environ['MPLCONFIGDIR']='/tmp/opd_matplotlib'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from decimal import Decimal,ROUND_HALF_UP
P=Path(__file__).parent
r=json.loads((P/'pass8_analysis.json').read_text())['metrics']
keys=['accuracy','pass_at_8','pass_all_8'];ml=['Mean acc','Pass@8','All 8 correct']
ds=list(r['accuracy']['datasets']);names=['BBH','CRUXEval-I','CRUXEval-O','HumanEval+','LiveCodeBench','MATH500','MBPP+','OlympiadBench','Overall']
values=np.array([[r[k]['datasets'][d] for k in keys] for d in ds]+[[r[k]['overall'] for k in keys]])
assert values.shape==(9,3,22) and np.all(values[:,2]<=values[:,0]+1e-9) and np.all(values[:,0]<=values[:,1]+1e-9)
def fmt(v):return str(Decimal(str(round(float(v),6))).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
fig,axes=plt.subplots(9,1,figsize=(23,13),layout='constrained')
cmap=plt.get_cmap('Blues');norm=Normalize(0,100)
for ax,name,V in zip(axes,names,values):
 ax.imshow(V,aspect='auto',cmap=cmap,norm=norm);ax.set_title(name,loc='left',fontweight='bold',fontsize=12,pad=6)
 ax.set_yticks(range(3),ml);ax.set_xticks(range(22),[str(i) for i in range(21)]+['Teacher']);ax.xaxis.tick_top();ax.tick_params(length=0,pad=5)
 for y in range(3):
  for x in range(22):ax.text(x,y,fmt(V[y,x]),ha='center',va='center',fontsize=9,color='white' if V[y,x]>=65 else '#15283b')
 ax.set_xticks(np.arange(-.5,22,1),minor=True);ax.set_yticks(np.arange(-.5,3,1),minor=True);ax.grid(which='minor',color='white',lw=1.5);ax.tick_params(which='minor',bottom=False,left=False,top=False);ax.axvline(20.5,color='#d07d24',lw=2)
 for s in ax.spines.values():s.set_visible(False)
fig.suptitle('Accuracy, Pass@8 and All-8 success by intervention budget (%)',fontsize=20,fontweight='bold')
fig.supxlabel('Shared color scale: 0% (light) to 100% (dark). 100 problems/dataset × 8 rollouts. Teacher starts from the original prompt; intervention runs share a frozen student prefix.',fontsize=11)
fig.savefig(P/'metric_tables_all_caps.png',dpi=180);fig.savefig(P/'metric_tables_all_caps.pdf');plt.close(fig)
with (P/'metric_tables_long.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['dataset','metric']+[str(i) for i in range(21)]+['Teacher'])
 for name,V in zip(names,values):
  for metric,row in zip(ml,V):w.writerow([name,metric]+[fmt(x) for x in row])
data={'names':names,'metrics':ml,'values':values.tolist()}
html='''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>逐数据集指标对比</title>
<style>body{font:15px system-ui,sans-serif;margin:32px;color:#193047;background:#fafcfe}h1{font-size:26px}p{color:#526579;line-height:1.6}.controls{display:flex;gap:16px;align-items:center;margin:22px 0}select{font:inherit;padding:8px;border:1px solid #bbcbd9;border-radius:6px;background:white}section{margin:24px 0}h2{font-size:18px;margin:10px 0}.scroll{overflow:auto}table{border-collapse:separate;border-spacing:2px;width:100%;font-variant-numeric:tabular-nums}td,th{padding:12px 9px;text-align:center;white-space:nowrap}th{font-size:13px}th:first-child{text-align:left;min-width:100px}.teacher{border-left:3px solid #d07d24}footer{margin-top:28px;color:#526579}.legend{display:inline-block;width:140px;height:12px;background:linear-gradient(90deg,rgb(244,249,253),rgb(25,93,155));margin:0 8px}</style>
<h1>各数据集：平均 acc · pass@8 · pass all 8</h1>
<p>单位：%。pass@8 = 八次至少一次正确；pass all 8 = 八次全部正确。每个数据集 100 题，每题 8 次 rollout。</p>
<div class="controls"><label>介入预算列 <select id="mode"><option value="key">关键档位：0 / 5 / 10 / 15 / 20</option><option value="all">完整档位：0–20</option></select></label><span>0%<span class="legend"></span>100%</span></div>
<main id="tables"></main>
<footer>Teacher 从原始题目生成；介入实验保留同一 student prefix。总体为八个数据集等权平均。色阶统一为 0–100%，数字保留两位小数。</footer>
<script id="data" type="application/json">DATA_PAYLOAD</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent());const mode=document.getElementById('mode');const tables=document.getElementById('tables');
function draw(){const cols=mode.value==='all'?[...Array(21).keys(),21]:[0,5,10,15,20,21];tables.replaceChildren();data.names.forEach((name,i)=>{const section=document.createElement('section');const h=document.createElement('h2');h.textContent=name;section.append(h);const wrap=document.createElement('div');wrap.className='scroll';const table=document.createElement('table');table.setAttribute('aria-label',name+'三项指标');const head=table.createTHead().insertRow();['指标',...cols.map(c=>c===21?'Teacher':c+' 次')].forEach(label=>{const th=document.createElement('th');th.textContent=label;head.append(th)});const body=table.createTBody();data.metrics.forEach((metric,j)=>{const tr=body.insertRow();const label=document.createElement('th');label.scope='row';label.textContent=metric;tr.append(label);cols.forEach(c=>{const v=data.values[i][j][c];const td=tr.insertCell();td.textContent=(Math.round((v+1e-9)*100)/100).toFixed(2);const t=v/100;td.style.backgroundColor=`rgb(${244-219*t},${249-156*t},${253-98*t})`;td.style.color=v>=65?'white':'#193047';if(c===21)td.className='teacher';})});wrap.append(table);section.append(wrap);tables.append(section)})}mode.addEventListener('change',draw);draw();
</script></html>
'''
(P/'metric_tables.html').write_text(html.replace('DATA_PAYLOAD',json.dumps(data)))
print('Saved PNG, PDF, HTML and CSV; verified all 594 metric cells.')
