import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
rows=[]
N={'DeepMath':1000,'MATH500':500,'OlympiadBench':674,'BBH':1080,'AIME24':30,'AIME25':30}
BASE={'DeepMath':399,'MATH500':364,'OlympiadBench':202,'BBH':378,'AIME24':5,'AIME25':4}
MAP={'math500':'MATH500','olympiad_bench':'OlympiadBench','bbh':'BBH','aime24':'AIME24','aime25':'AIME25'}
def add(ds,method,k,tau,luna,correct,source,cap='',ban=True):
 n=N[ds]; a=correct/n; b=BASE[ds]/n
 rows.append(dict(benchmark=ds,method=method,teacher_k=k,gap_rule='absolute' if tau else '',threshold=tau,luna=luna,max_takeovers=cap,ban_student_entry=ban,questions=n,correct=correct,accuracy_percent=a*100,delta_pp_vs_student=(a-b)*100,relative_gain_percent=(a/b-1)*100,metric='single_rollout_accuracy',source=source))
def source(path,mode,method,k='',tau='',luna=False,cap='',ban=True):
 p=ROOT/path
 d=json.loads(p.read_text())[mode]
 for key,v in d.items():
  assert v['questions']==N[MAP[key]]
  add(MAP[key],method,k,tau,luna,v['correct'],path,cap,ban)
hard='experiments/hard_three_20260911/artifacts/summary.json'
for mode in ['student','teacher']:source(hard,mode,mode,ban=False)
source(hard,'online','luna_k8',8,2,True)
mathp='experiments/math500_three_20260911/results/summary.json'
d=json.loads((ROOT/mathp).read_text())
for mode,method,k,tau,luna in [('student','student','','',False),('online','luna_k8',8,2,True),('prefix256','prefix256',256,'',False)]:add('MATH500',method,k,tau,luna,d[mode]['correct'],mathp,ban=mode=='online')
source('experiments/hard_k32_20260911/artifacts/summary.json','online','luna_k32',32,2,True)
source('experiments/threshold_k8_nojudge_20260911/artifacts/summary.json','online','no_luna_t2',8,2)
source('experiments/threshold3_mathteacher_20260911/artifacts/summary.json','online','no_luna_t3',8,3)
source('experiments/threshold3_mathteacher_20260911/artifacts/summary.json','teacher','teacher',ban=False)
source('experiments/random_k8_max8_20260911/artifacts/summary.json','online','random_cap8',8,'',False,8)
for method,corr in [('student',399),('teacher',610),('luna_k8',476),('luna_k32',504),('random_cap8',426)]:
 k={'luna_k8':8,'luna_k32':32,'random_cap8':8}.get(method,'')
 add('DeepMath',method,k,2 if 'luna' in method else '', 'luna' in method,corr,'experiments/online_comparison_20260911/comparison.json' if method in ['student','teacher','luna_k8'] else 'remote:opd_eval_deepmath/'+('online_chunk64_k32_test1000_h100_seed20260911' if method=='luna_k32' else 'random_k8_cap8_test1000_h100_seed20260911')+'/results/summary.json',8 if method=='random_cap8' else '',method.startswith('luna'))
for k,c in [(32,417),(64,414),(128,424),(256,400)]:add('DeepMath',f'prefix{k}',k,'',False,c,'normalize_single_prefix.py: remote teacher_prefix_handoff_20260909/h%d.json'%k,ban=False)
assert len({(r['benchmark'],r['method']) for r in rows})==len(rows)
with (OUT/'single_rollout_results.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'single_rollout_results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
lookup={(r['benchmark'],r['method']):r for r in rows}
labels={'student':'Student直接到底','teacher':'Teacher直接到底','prefix32':'Teacher开头32→Student到底','prefix64':'Teacher开头64→Student到底','prefix128':'Teacher开头128→Student到底','prefix256':'Teacher开头256→Student到底','luna_k8':'无限介入，Luna，绝对差>2，k=8','luna_k32':'无限介入，Luna，绝对差>2，k=32','no_luna_t2':'无限介入，无Luna，绝对差>2，k=8','no_luna_t3':'无限介入，无Luna，绝对差>3，k=8','random_cap8':'随机位置，k=8，最多8次（额外对照）'}
ds=['DeepMath','MATH500','OlympiadBench','BBH','AIME24','AIME25']
s=['# 当前介入实验总对比（2026-09-11）','','范围：原始Qwen3-1.7B Student与Qwen3-14B Teacher的推理介入实验。最后一类按“无Luna判定”解释。主表全部non-thinking、每题单次回答正确率，不是pass@8/64。未混入4B学生、训练后checkpoint或thinking模式基线。','','## 1. 完成实验：单次回答正确率','', '| 方法 | '+' | '.join(f'{x} (n={N[x]})' for x in ds)+' |','|---|'+'---:|'*len(ds)]
for method,label in labels.items():
 vals=[]
 for x in ds:
  r=lookup.get((x,method));vals.append('—' if r is None else f"{r['accuracy_percent']:.2f}% ({r['correct']}/{r['questions']})")
 s.append('| '+label+' | '+' | '.join(vals)+' |')
s += ['','—表示本次盘点未找到该配置的完成结果，不代表0%。已跑但未评分的配置另列。BBH为官方27个子任务各40题、共1080题；OlympiadBench为英文数学OE_TO_maths_en_COMP全674题。AIME每年仅30题。','',
'所有上表阈值介入都是历史绝对差 `abs(logp_student−logp_teacher)>τ`，不是后来更正的正向差规则。在线student chunk=64；Luna只判断候选是否接管，不生成最终答案、不担任答案评分器。历史在线组保留teacher首token不得等于student原token的规则。','',
'DeepMath统一布尔答案归一化：原student/teacher单次28.3%/43.6%已统一为39.9%/61.0%。本次只读重汇总了开头接管单次实验：k32/64/128/256旧分数29.3/28.3/29.3/27.6%，统一后41.7/41.4/42.4/40.0%；分别新增124/131/131/124个正确、没有反向改判。原输出和原评分未改写。新规则不是对证明过程的审核。','',
'DeepMath直接/开头接管的历史seed为20260811，在线seed为20260911；其余主要20260911，temperature=1/top_p=1。阈值3实验用H100，许多基线用H800。单次结果不能据此声称仅一个因素的严格因果差异。','',
'k32的OlympiadBench与BBH各有1条被用户提前停止，按错误纳入原分母；与其它组跑到既定结束条件的协议不同。随机DeepMath组允许原token，其它五个benchmark的随机组强制首token不同，两组随机结果不能当作同一协议。','',
'## 2. 单次anchor支持：同一DeepMath 958题子集','',
'1000题中958题找到了离线anchor；剩余42题未纳入这张表，不能把43.33%直接放进全1000题排行榜。Teacher开头与单次anchor均为8个teacher分支×8个student continuation，每题64个回答；Student直接为64个回答，Teacher直接为8个回答。以下为回答平均正确率。','',
'| 方法 | 回答数 | 正确率 | 相对同子集Student |','|---|---:|---:|---:|']
a=json.loads((ROOT/'experiments/anchor_comparison_20260910/comparison.json').read_text())
for key in ['student64','teacher','h32','h64','h128','h256','anchor']:
 v=a['same_subset'][key];label={'student64':'Student直接×64','teacher':'Teacher直接×8','anchor':'单次anchor处Teacher256→Student到底'}.get(key,'Teacher开头'+key[1:]+'→Student到底')
 s.append(f"| {label} | {v['samples']:,} | {v['accuracy']*100:.4f}% | {(v['accuracy']-a['same_subset']['student64']['accuracy'])*100:+.4f} pp |")
s += ['','同题子集中，anchor256相对开头256提高1.5576个百分点，相对Student直接提高3.8769个百分点。它使用原始student前缀、anchor筛选和分支采样成本，不能视为等算力比较。','',
'全1000题的多次采样平均正确率（同一新评分口径）：','', '| 方法 | 回答数 | 正确率 |','|---|---:|---:|']
for key,v in a['full1000'].items():s.append(f"| {key} | {v['samples']:,} | {v['accuracy']*100:.4f}% |")
s += ['','单次1×1与8×8结果须分开：例如开头256在单次实验40.0%，在8×8采样平均42.9328%。不能只挑更高者或把pass@64当正确率。','',
'## 3. 更早的train25已选anchor集合：支持长度实验','',
'此表为13,244个已选prefix上的continuation实验，不是held-out1000题。Student基线从同一个anchor前缀继续，不是从问题开头直接rollout。每个bridge长度847,616条continuation，基线105,952条。此处保留既有报告评分口径，不直接合并到上述布尔归一化测试表。','',
'| 条件 | 回答平均正确率 | 相对prefix Student |','|---|---:|---:|',
'| 同prefix Student到底 | 39.2668% | — |','| anchor处Teacher32→Student | 40.4995% | +1.2326 pp |','| anchor处Teacher64→Student | 41.0011% | +1.7343 pp |','| anchor处Teacher128→Student | 42.0167% | +2.7498 pp |','| anchor处Teacher256→Student | 44.2065% | +4.9396 pp |','| 同prefix Teacher到底 | 57.3118% | +18.0450 pp |','',
'长度增加的主要收益是提高回答成功概率和重复成功稳定性；其pass@8约69.8%–70.0%，不应解释为解题覆盖随k同幅增加。','',
'## 4. 有无Luna与阈值的teacher用量','',
'下列每题teacher保留token不是总计算量，不含候选logit评分、丢弃student草稿和Luna请求。','',
'| Benchmark | Luna τ2 k8平均介入次数 | 无Luna τ2 k8平均次数 | 有Luna平均teacher token | 无Luna平均teacher token |','|---|---:|---:|---:|---:|',
'| MATH500 | 2.25 | 9.25 | 17.99 | 73.85 |','| OlympiadBench | 4.54 | 16.10 | 36.29 | 128.03 |','| BBH | 3.31 | 10.49 | 25.49 | 80.66 |','| AIME24 | 8.17 | 27.67 | 65.33 | 221.10 |','| AIME25 | 6.27 | 23.90 | 50.13 | 191.20 |','',
'DeepMath Luna k8/k32分别平均5.896/5.042次介入、47.156/160.817个teacher token。MATH500固定开头256平均251.96个teacher token，在线Luna k8平均17.99个，但后者另付候选评分与Luna成本。','',
'无Luna τ3 k8：MATH/Olympiad/BBH平均介入3.924/6.789/8.217次，teacher汇总token占比3.80%/3.70%/23.22%；τ2对应8.56%/8.60%/29.72%。阈值3节省介入，但MATH/Olympiad正确率降低1.80/3.71个百分点，BBH增加0.93个百分点；不存在所有benchmark一致最优阈值。','',
'## 5. 尚未有完整评分的配置（远程18:16左右快照）','',
'| 配置 | 数据集 | 当前证据 |','|---|---|---|',
'| 绝对差>2 + Luna，k1，不限次数，首token排除 | MATH500/Olympiad/BBH，共2254题 | GPU102597运行，1542/2254生成、0评分；不能填正确率 |',
'| 正向差student−teacher>2 + Luna，k8，最多8次 | 同2254题 | GPU102611运行，502/2254生成；0评分 |',
'| 正向差>2，无Luna，k8，最多8次 | 同2254题 | 已安排，顺序等待前组 |',
'| 正向差>2 + Luna，k8，不限次数，允许原token | 同2254题 | 已安排，顺序等待前组 |',
'| 随机Bernoulli介入，k8，不限次数，允许原token | 同2254题 | 已安排，按各数据集历史Luna平均间隔设概率；仅期望频率匹配 |','',
'未找到完成结果：无Luna k32及其它k支持长度扫点、有Luna τ3扫点、MATH500有Luna k32、DeepMath无Luna阈值扫点、其它benchmark的单次anchor与开头k长度扫点。不得把最多8次的组归入无限次数，亦不得把正向差新实验与绝对差历史实验混为一谈。','',
'## 6. 可以支持的结论','',
'1. 完成的在线介入中，无Luna τ2 k8在MATH500/OlympiadBench/BBH取得84.20%/46.14%/75.74%，比有Luna k8高2.80/11.72/30.93 pp；teacher用量同时增加，尚不能解释为等预算下Luna有害。',
'2. Luna k32相对k8在DeepMath/Olympiad/BBH提高2.8/1.48/7.78 pp，teacher写入量更高；AIME波动且AIME25下降，不能宣称所有benchmark受益。',
'3. 单次anchor256在相同958题、相同64回答结构下比开头256高1.56 pp；这支持介入位置有价值，但不是等计算预算因果消融。',
'4. 当前结果未超过相应Teacher直接rollout：BBH无Luna τ3接近Teacher（76.67 vs77.31），Olympiad无Luna τ2接近Teacher（46.14 vs46.59）。单次小样本不支持稳定等价声明。',
'5. 优先完成正在运行的正向差/首token规则消融，再做同teacher预算的Luna开关对照与多seed；AIME需更多采样。当前KV复用只有功能/性能验证，没有新全量正确率实验，不能把其加速数混入上述准确率证据。','',
'## 来源与可复核产物','',
'- single_rollout_results.csv / .json：每个完成配置的正确数、分母、阈值、次数上限、相对提升、源路径。',
'- normalize_single_prefix.py：只读重汇总旧单次开头介入的脚本；仅采用现有归一化函数，不改原输出。',
'- experiments/anchor_comparison_20260910/{comparison.json,SCORING_AUDIT.md}：多次采样同题与评分口径。',
'- experiments/{hard_three,hard_k32,threshold_k8_nojudge,threshold3_mathteacher,random_k8_max8}_20260911/artifacts/summary.json。',
'- experiments/math500_three_20260911/results/summary.json；experiments/online_comparison_20260911/comparison.json。',
'- analysis/prefix_bridge_teacher_average_accuracy_20260908.csv：train25前缀支持力度。',
'- experiments/{threshold2_luna_k1,takeover_ablation}_20260911/STATUS.md与本次远程只读快照。']
(OUT/'REPORT.md').write_text('\n'.join(s)+'\n')
print('rows',len(rows),'report',OUT/'REPORT.md')
