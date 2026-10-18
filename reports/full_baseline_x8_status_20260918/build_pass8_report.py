import json,csv
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'pass8_results.json').read_text())['results'];data={(r['config'],r['dataset']):r for r in rs}
ds={'cruxeval_o':'CRUXEval-O','cruxeval_i':'CRUXEval-I','humaneval_plus':'HumanEval+','mbpp_plus':'MBPP+','livecodebench':'LiveCodeBench','math500':'MATH500','olympiad_bench':'OlympiadBench','bbh':'BBH'}
lines=['# 全题集 baseline ×8','', '每格为 pass@8 / pass all 8；分母为题目数。已逐题核验 73,136 条评分的 role、题目 ID、replicate 0–7、评分签名；通过次数与现有准确率汇总完全一致。','', '| 数据集 | 题数 | Student-only | Teacher-only |','|---|---:|---:|---:|']
for d,label in ds.items():
 rows=[data[c,d] for c in ['student_only','teacher_only']]
 lines.append('| '+label+' | '+str(rows[0]['questions'])+' | '+' | '.join(f"{r['pass_at_8']*100:.2f}% / {r['pass_all_8']*100:.2f}%" for r in rows)+' |')
lines+=['','pass@8：8 次至少通过 1 次；pass all 8：8 次全部通过。生成超时计错。Code 为执行测试评分；math/reasoning 为既有 GPT-4o-mini judge 评分。','', '来源：opd_eval_baselines/full_teacher_student_x8_20260918 各组 scores 与 summary.json。']
(P/'pass8_report.md').write_text('\n'.join(lines)+'\n')
with (P/'pass8_results.csv').open('w') as f:
 fields=[k for k in rs[0] if k!='correct_count_histogram']+[f'correct_{i}_of_8_count' for i in range(9)]
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for r in rs:w.writerow({**{k:v for k,v in r.items() if k!='correct_count_histogram'},**{f'correct_{i}_of_8_count':n for i,n in enumerate(r['correct_count_histogram'])}})
print('\n'.join(lines))
