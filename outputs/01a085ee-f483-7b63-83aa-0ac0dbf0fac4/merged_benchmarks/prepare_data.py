import csv, json, math, collections
from pathlib import Path

ROOT = Path('/Users/leolee/Documents/OPD')
OUT = Path(__file__).parent
datasets = ['math500','olympiad_bench','bbh','cruxeval_o','cruxeval_i','humaneval_plus','mbpp_plus','livecodebench']
labels = ['MATH500','OlympiadBench','BBH','CRUXEval-O','CRUXEval-I','HumanEval+','MBPP+','LiveCodeBench']
ns = dict(zip(datasets,[500,674,1080,800,800,164,378,175]))
cfgs = ['student_only','teacher_only']+[f'k{k}_t{t:g}' for t in [2,2.5,3,3.5,4] for k in [1,4,8,16,32]]
records = {}
def add(g,ds,n,correct,acc,events,share,length,timeouts,policy,source):
    assert n==ns[ds] and abs(correct/n-acc)<1e-10
    assert 0<=share<=1 and length>0 and events>=0
    assert (g,ds) not in records
    records[g,ds]=dict(config=g,dataset=ds,n=n,correct=correct,acc=acc,mean_interventions=events,teacher_token_fraction=share,mean_output_tokens=length,generation_timeouts=timeouts,stop_policy=policy,source=source)

code=ROOT/'experiments/code_reasoning_grid_20260913/reports/FINAL_COMPARISON.csv'
for x in csv.DictReader(code.open()):
    add(x['config'],x['dataset'],int(x['n']),int(x['correct']),float(x['acc']),float(x['mean_interventions']),float(x['teacher_token_fraction']),float(x['mean_output_tokens']),int(x['timeouts']),'900秒累计生成截止；超时保留并计错','C：代码最终汇总')

agg=json.loads((OUT/'math_metrics.json').read_text())
math_csv=ROOT/'experiments/nojudge_grid_20260912/reports/FINAL_COMPARISON.csv'
for x in csv.DictReader(math_csv.open()):
    t=float(x['threshold']);k=int(x['k']);g=f'k{k}_t{t:g}';old=g.replace('.', 'p');sum_timeout=0
    for ds in datasets[:3]:
        a=agg[old][ds];n=a['n'];acc=float(x[ds+'_acc_percent'])/100
        ev=float(x[ds+'_mean_interventions']);share=float(x[ds+'_teacher_token_percent'])/100
        assert abs(ev-a['events']/n)<1e-10 and abs(share-a['teacher']/a['tokens'])<1e-10
        policy='原生成上限；未新增900秒截止'
        if k==32:policy='900秒累计生成截止；超时保留并计错'
        elif k==16 and t==4:policy='混合：本组373条旧输出、1881条启用900秒截止'
        add(g,ds,n,round(acc*n),acc,ev,share,a['tokens']/n,a['timeouts'],policy,'M：数学网格最终汇总及token实测')
        sum_timeout+=a['timeouts']
    assert sum_timeout==int(x['timeout_count'])

hroot=ROOT/'experiments/hard_three_20260911/artifacts'
hsummary=json.loads((hroot/'summary.json').read_text())
for role in ['student','teacher']:
    lengths=collections.defaultdict(list)
    for f in (hroot/role/'completed').glob('*.json'):
        x=json.loads(f.read_text())
        if x['dataset'] in datasets[:3]:lengths[x['dataset']].append(len(x['response_token_ids']))
    for ds in ['olympiad_bench','bbh']:
        x=hsummary[role][ds];ls=lengths[ds];assert len(ls)==x['questions']
        add(role+'_only',ds,x['questions'],x['correct'],x['accuracy'],0,int(role=='teacher'),sum(ls)/len(ls),0,'历史直接生成基线；无900秒截止','H：hard_three原始基线及summary')
m=json.loads((ROOT/'experiments/math500_three_20260911/results/summary.json').read_text())['student']
add('student_only','math500',500,m['correct'],m['accuracy'],0,0,m['mean_tokens'],0,'历史直接生成基线；无900秒截止','S：math500_three基线summary')
mt=json.loads((OUT/'math_teacher.json').read_text());m=mt['summary']['teacher']['math500']
add('teacher_only','math500',500,m['correct'],m['accuracy'],0,1,mt['mean_tokens'],0,'历史直接生成基线；无900秒截止','T：threshold3_mathteacher直接基线')
rows=[records[g,ds] for g in cfgs for ds in datasets]
assert len(rows)==216 and len(records)==216
assert sum(r['n'] for r in rows)==123417
payload=dict(datasets=datasets,labels=labels,configs=cfgs,rows=rows)
(OUT/'data.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
with (OUT/'全部实验_明细.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
metrics=['acc','mean_interventions','teacher_token_fraction','mean_output_tokens']
with (OUT/'全部实验_总表.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['实验','阈值','k']+[f'{label}_{m}' for label in labels for m in metrics])
    for g in cfgs:
        t=None if g.endswith('_only') else float(g.split('_t')[1]);k=None if t is None else int(g.split('_')[0][1:])
        w.writerow([g,t,k]+[records[g,ds][m] for ds in datasets for m in metrics])
md=['# 数学、推理与代码基准总表','', '**每格依次：准确率 / 平均介入次数 / Teacher token占比 / 平均输出token数。**','',
    'Student为Qwen3-1.7B，Teacher为Qwen3-14B；单次non-thinking，temperature=1、top_p=1。无Luna组按有符号正阈值触发，介入次数不限。','',
    '| 实验 | 阈值 | k | '+' | '.join(labels)+' |','|'+'---|'*11]
for g in cfgs:
    t='—' if g.endswith('_only') else g.split('_t')[1];k='—' if t=='—' else g.split('_')[0][1:]
    cells=[]
    for ds in datasets:
        r=records[g,ds];cells.append(f"{r['acc']:.2%} / {r['mean_interventions']:.2f} / {r['teacher_token_fraction']:.2%} / {r['mean_output_tokens']:.2f}")
    md.append('| '+' | '.join([g,t,k]+cells)+' |')
md+=['','- 题数：MATH500 500；OlympiadBench英文纯文本数学674；BBH 27×40=1080；CRUXEval-O/I各800；HumanEval+164；MBPP+378；LiveCodeBench v6增量175。',
 '- Teacher占比按输出token总量加权，不包括prompt或丢弃草稿。平均次数包含零介入题目；长度包含超时保留输出。Teacher-only占比100%、介入次数0，表示直接生成而非接管。',
 '- 停止规则不同：数学网格前19组无新增900秒截止；k16/阈值4混合373条旧输出和1881条限时输出；k32五组均限时。代码全部27组均限时900秒。数学历史基线无900秒截止。',
 '- 数学历史基线与代码基线采用不同生成组织方式（代码为64-token分段）。不同数据集不合并为一个总准确率。',
 '- 代码评分包含生成超时计错、代码执行超时和进程崩溃计错；明细“generation_timeouts”仅统计生成超时。',
 '- 来源：数学网格FINAL_COMPARISON.csv及原始token统计；hard_three、math500_three和threshold3_mathteacher历史基线；代码网格FINAL_COMPARISON.csv。']
(OUT/'全部实验_总表.md').write_text('\n'.join(md),encoding='utf-8')
print('Validated 216 records,27 configurations,8 datasets;123417 evaluated traces. All math token shares match source.')
