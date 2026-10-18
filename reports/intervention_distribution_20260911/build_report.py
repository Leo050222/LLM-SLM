import json,csv
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'deepmath_summary.json').read_text())+json.loads((P/'local_summary.json').read_text())
order=['deepmath','math500','olympiad_bench','bbh','aime24','aime25']
configs=['Luna_abs2_k8','Luna_abs2_k32','NoLuna_abs2_k8','NoLuna_abs3_k8']
rs.sort(key=lambda r:(order.index(r['dataset']),configs.index(r['config'])))
names={'deepmath':'DeepMath','math500':'MATH500','olympiad_bench':'OlympiadBench','bbh':'BBH','aime24':'AIME24','aime25':'AIME25'}
labels={'Luna_abs2_k8':'Luna τ2 k8','Luna_abs2_k32':'Luna τ2 k32','NoLuna_abs2_k8':'无Luna τ2 k8','NoLuna_abs3_k8':'无Luna τ3 k8'}
def table(headers,rows):return ['| '+' | '.join(headers)+' |','|'+'---|'*len(headers)]+['| '+' | '.join(map(str,r))+' |' for r in rows]
def dump(name,rows):
 with (P/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
freq=[];pos=[]
for r in rs:
 freq.append(dict(dataset=r['dataset'],config=r['config'],questions=r['questions'],takeovers=r['takeovers'],mean_takeovers=r['counts']['mean'],median_takeovers=r['counts']['p50'],p90_takeovers=r['counts']['p90'],max_takeovers=r['counts']['max'],zero_takeovers=r['zero'],mean_output_tokens=r['length']['mean'],mean_teacher_tokens=r['teacher_tokens_mean'],pooled_teacher_share=r['teacher_share'],student_tokens_per_takeover=r['pooled_student_tokens_per_takeover'],median_student_segment_before_takeover=r['student_segment_before_takeover']['p50'],median_between_bridge_student_gap=r['actual_between_bridge_student_gap']['p50'],median_first_position=r['first_position']['p50']))
 for i,n in enumerate(r['relative_position_decile_counts']):pos.append(dict(dataset=r['dataset'],config=r['config'],decile_start=i/10,decile_end=(i+1)/10,count=n,event_weighted_percent=r['relative_position_decile_percent'][i],active_trace_weighted_percent=r['trace_weighted_decile_percent'][i]))
 assert sum(r['relative_position_decile_counts'])==r['takeovers']
 assert sum(r['count_bins'].values())==r['questions']
dump('frequency.csv',freq);dump('relative_positions.csv',pos);(P/'all_summary.json').write_text(json.dumps(rs,indent=2))
s=['# 介入频率与位置分布整理（2026-09-11）','','使用已完成、已保存的单次rollout轨迹重新聚合，未启动新生成。本次21个benchmark×配置组，合计10,756条轨迹。Qwen3-1.7B Student / Qwen3-14B Teacher，non-thinking，student chunk64，历史绝对差阈值；这些组均不设接管次数上限。k32的Olympiad与BBH各包含一条用户中止的部分轨迹，其位置以已保留长度归一化。随机对照因次数上限和位置抽样机制不同，不混入以下主分布。','','## 口径','',
'- 介入次数：实际生成非空teacher bridge的事件数。没有通过Luna的候选不算介入。','- 相对位置：teacher bridge起点（从0开始）/该条最终保留回答总token数；分母包含student和teacher，不含prompt或丢弃草稿。','- 每次介入前student段：第一次从回答开始算，其余从前一teacher bridge结束算。严格相邻介入间隔则不含第一次。','- 全局密度：全体保留student token数/介入总数；包含回答尾段及零介入题，不是实测相邻间隔均值。','- 位置默认按介入事件加权，每次介入算一个；CSV另给每条有介入trace等权的分布，减少高介入长轨迹的权重。','- teacher占比为teacher保留token/所有保留token，不是GPU成本占比。','','## 1. 介入频率','']
s+=table(['Benchmark','配置','N','总次数','均值','中位数','P90','最大','0次题数'],[[names[r['dataset']],labels[r['config']],r['questions'],r['takeovers'],f"{r['counts']['mean']:.2f}",f"{r['counts']['p50']:g}",f"{r['counts']['p90']:.1f}",r['counts']['max'],r['zero']] for r in rs])
s+=['','## 2. 间隔、首次介入位置和token占比','']
s+=table(['Benchmark','配置','全局student token/次','介入前student段中位数','严格相邻间隔中位数','首次介入位置中位数','平均teacher token','teacher占比'],[[names[r['dataset']],labels[r['config']],f"{r['pooled_student_tokens_per_takeover']:.2f}",f"{r['student_segment_before_takeover']['p50']:g}",f"{r['actual_between_bridge_student_gap']['p50']:g}",f"{r['first_position']['p50']:g}",f"{r['teacher_tokens_mean']:.2f}",f"{100*r['teacher_share']:.2f}%"] for r in rs])
s+=['','首次位置仅针对有介入的题。BBH无Luna两组首次位置中位数为0，表示至少半数有介入轨迹在回答的第一个token处就交给teacher，不是第一个64-token块结束后才开始接管。','','## 3. 实际介入相对位置分布（事件占比）','']
s+=table(['Benchmark','配置']+[f'{10*i}–{10*(i+1)}%' for i in range(10)],[[names[r['dataset']],labels[r['config']]]+[f'{v:.2f}%' for v in r['relative_position_decile_percent']] for r in rs])
s+=['','## 4. DeepMath次数档位与chunk内部位置','']
dm=[r for r in rs if r['dataset']=='deepmath']
s+=table(['每题介入次数','k8题数','k32题数'],[[b,dm[0]['count_bins'][b],dm[1]['count_bins'][b]] for b in ['0','1-2','3-5','6-10','11+']])
s+=['']+table(['chunk内偏移（0-based）','k8事件占比','k32事件占比'],[[f'{i*16}–{i*16+15}',*[f"{r['chunk_offset_quartile_counts'][i]/r['takeovers']*100:.2f}%" for r in dm]] for i in range(4)])
s+=['',f"DeepMath k8/k32平均chunk内偏移分别{dm[0]['chunk_offset']['mean']:.2f}/{dm[1]['chunk_offset']['mean']:.2f}；有介入的chunk占比{dm[0]['intervened_round_share']*100:.2f}%/{dm[1]['intervened_round_share']*100:.2f}%。偏移分布受逐个候选检查、选第一个被接受候选的策略影响，不能仅据此认定推理分岔在chunk前部自然更多。",'',
'## 5. 历史候选token分布：不等于实际接管','',
'此前DeepMath原始1000条student trace、1,303,870个token中，24,249个（1.860%）满足正向差 logp_S−logp_T>2。这里是正向差候选，前文实际在线介入采用绝对差再经Luna或直接触发；轨迹也已被介入改变，不可混为同一总体。','']
old=[8.55,12.57,12.53,11.80,10.79,9.95,9.59,8.66,8.19,7.37]
s+=table(['回答相对位置','正向差>2候选token占比','实际Luna k8介入占比','实际Luna k32介入占比'],[[f'{i*10}–{(i+1)*10}%',f'{old[i]:.2f}%',*[f"{r['relative_position_decile_percent'][i]:.2f}%" for r in dm]] for i in range(10)])
s+=['','候选事件占比在10–20%区间最高（12.57%），20–30%紧随（12.53%）；如果统计各位置区间内的候选密度（逐trace等权），最高是20–30%（2.77%）。前半段含56.24%的候选事件。此前“20–30%最高”说的是密度，不应误写成候选事件数最高。候选报告的正确/错误分组使用旧评分，此处不沿用其正确率结论。','','## 6. 结论与后续用途','',
'1. DeepMath k8→k32，平均介入5.896→5.042，介入前student段中位数96→108；teacher保留token47.156→160.817。更长支持降低次数，但teacher token明显增加。',
'2. Luna筛选后的实际介入分布覆盖大部分回答；DeepMath两端10%明显少于中间各段，不支持只在开头提供帮助。',
'3. 去掉Luna后，介入明显变密、首次介入更早。BBH无Lunaτ2的介入前student段中位数只有7token；MATH/Olympiad分别39/38token。',
'4. 阈值2→3减少无Luna介入频率；BBH仍然密集，说明不同benchmark的触发密度差异很大，不宜用统一“每隔多少token”描述全部任务。',
'5. 归一化位置依赖最终输出长度（它本身受介入影响），这里只作描述。若用于自适应chunk策略，应先在独立开发集验证；不能直接把teacher输出占比当成计算节省。','','## 来源','',
'- experiments/hard_three_20260911/INTERVENTION_ANALYSIS.md：历史频率、长度和次数分组。',
'- experiments/threshold_k8_nojudge_20260911/LUNA_LENGTH_TOKEN_SHARE.md：Luna开关token占比。',
'- experiments/takeover_ablation_20260911/random_rates.json：384.34/326.43/86.00的全局student token/次统计。',
'- reports/deepmath_nonthinking_gt2_analysis.md：历史正向差候选位置分布。',
'- aggregate.py：从上述已保存completed输出重新聚合实际位置；本地8756条，远程DeepMath2000条。',
'- all_summary.json、frequency.csv、relative_positions.csv：完整数据，包含频率分位数、位置分位数和两种加权口径。']
(P/'REPORT.md').write_text('\n'.join(s)+'\n')
print('Validated',len(rs),'groups and',sum(r['questions'] for r in rs),'traces')
