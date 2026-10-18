from common import *
import html
protocol='All configurations and baselines use eight rollouts per question, with the same question counts within each dataset. Student: Qwen3-1.7B; Teacher: Qwen3-14B. Math/reasoning replicate 0 differs from later replicates in judge/extraction and timeout protocol; deployment concurrency also varied. Generation timeouts are counted as incorrect. Results are descriptive, with no confidence intervals or significance claims.'
captions={
'accuracy_teacher_share':'Mean accuracy versus Teacher token share for all 25 intervention configurations on each dataset. Colors denote thresholds and marker shapes denote k; lines connect points in increasing k, which need not increase token share. Horizontal lines show x8 Student-only and Teacher-only accuracy references, not their token-share coordinates (0% and 100%, respectively). Circled and annotated points are k1_t2 and all configurations attaining the observed maximum accuracy. Axes are scaled independently across panels. Teacher token share is the ratio of total Teacher output tokens to total output tokens; it excludes prompt, discarded drafts and Teacher scoring overhead and is not a compute-cost or speedup measure.',
'accuracy_teacher_share_zoom':'Detail view of the accuracy–Teacher token tradeoff, with independently zoomed vertical axes. Student-only accuracy lies below every panel and is stated numerically; Teacher-only is a horizontal reference. All data, color/marker encodings and selected annotations match the full-range figure. Small observed differences do not imply statistical significance.',
'stability_t2':'Per-question outcomes over eight rollouts for Student-only, Teacher-only and all five k values at threshold 2. Gray denotes zero correct rollouts (100 minus pass@8); light blue denotes one to seven correct rollouts (pass@8 minus pass all 8); dark blue denotes eight correct rollouts (pass all 8). Bars sum to 100%; right-hand values give mean accuracy over all outputs. Segment labels are percentages and are omitted below 8% for readability. The subset is fixed by threshold, not selected by each dataset’s best accuracy.',
'accuracy_heatmap':'Mean accuracy (%) across the full threshold-by-k grid. Math/reasoning occupy the top row; code datasets occupy the remaining rows. Each panel uses its own labeled color scale; color intensity is not comparable across datasets. Orange outlines mark every maximum computed before rounding, including ties. Student-only and Teacher-only x8 accuracies appear beneath each panel.',
'interventions_heatmap':'Mean number of Teacher interventions per generated output across the threshold-by-k grid, including zero-intervention and timeout samples. Both standalone baselines have zero interventions. Each panel has its own labeled color scale. Panel layout matches the accuracy heatmap.',
'length_ratio_heatmap':'Mean output length divided by the same dataset’s x8 Student-only mean output length. Ratios are computed from unrounded totals and include timeout samples; values below one indicate shorter outputs. Student baseline token lengths and Teacher/Student length ratios appear beneath each panel. Each panel has its own labeled color scale, so compare printed ratios across datasets.'}
tex=[]
for name,caption in captions.items():
 full=caption+' '+protocol
 tex.append('\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{figures/main_grid_x8_20260920/'+name+'.pdf}\n\\caption{'+full.replace('%','\\%').replace('_','\\_').replace('×',r'$\times$')+'}\n\\label{fig:'+name.replace('_','-')+'}\n\\end{figure*}\n')
# Use the readable 2x2 print exports for the default manuscript snippets.
tex=[]
for name,caption in captions.items():
 caption=caption.replace('Math/reasoning occupy the top row; code datasets occupy the remaining rows. ', '').replace('Panel layout matches the accuracy heatmap.', 'Dataset ordering follows the accuracy heatmap.')
 for part in [1,2]:
  artifact=name+f'_print_part{part}'
  tex.append('\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{figures/main_grid_x8_20260920/'+artifact+'.pdf}\n\\caption{'+(caption+f' Part {part} of 2. '+protocol).replace('%','\\%').replace('_','\\_')+'}\n\\label{fig:'+artifact.replace('_','-')+'}\n\\end{figure*}\n')
(OUT/'latex_includes.tex').write_text('\n'.join(tex))
zh=['准确率与 Teacher token 占比','局部放大：比较配置差异','稳定性：0/8、部分通过、8/8','参数分析：准确率热力图','补充：平均介入次数','补充：相对输出长度']
items=[]
for (name,caption),title in zip(captions.items(),zh):
 printlinks = ''.join(f'<a href="{name}_print_part{j}.pdf">论文拆分版 {j}（PDF）</a> · ' for j in [1,2])
 items.append(f'<section><h2>{title}</h2><p><a href="{name}.pdf">矢量 PDF</a> · <a href="{name}.svg">SVG</a> · <a href="{name}.png">300 dpi PNG</a></p>{printlinks}<img src="{name}.svg"><p>{html.escape(caption)}</p></section>')
(OUT/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>主表可视化 · x8 rollout</title><style>body{max-width:1500px;margin:auto;padding:40px;font:16px/1.7 system-ui;color:#243645;background:#fafbfc}section{background:white;padding:28px;margin:28px 0;border:1px solid #dde3e8;border-radius:12px}img{width:100%}a{color:#0072b2}h1,h2{line-height:1.3}p{max-width:1200px}</style><h1>主表可视化 · 8 次 rollout</h1><p>8 个数据集 × 25 个配置 + Student/Teacher 各 8 次 rollout 基线。共 216 个数据集–配置记录。主图保留全部基线，局部放大版用于看清配置间差异；热力图将数学/推理与代码分组。</p><p>Teacher token share 不等同于算力成本或加速比。数学/推理首轮与后续轮次的评分与超时设置不完全一致，以下为描述性展示，不作显著性判断。</p>'+''.join(items)+'<section><h2>数据与复现</h2><p><a href="plot_data.csv">完整绘图数据</a> · <a href="validation.json">数据校验与源文件 SHA256</a> · <a href="latex_includes.tex">LaTeX 图注</a></p><p>'+protocol+'</p></section></html>')
(OUT/'README.md').write_text('''# 主表可视化（8 次 rollout）

打开 index.html 浏览全部六张图。每张均提供矢量 PDF、SVG、300 dpi PNG。

- accuracy_teacher_share：2×4 主图，包含 Student/Teacher 水平参考线。
- accuracy_teacher_share_zoom：局部放大版，Student 基线在轴外明确注明数值。
- stability_t2：两个基线 + t=2 下全部五个 k；灰 / 浅蓝 / 深蓝表示 0 / 1–7 / 8 次正确。
- accuracy_heatmap：阈值 × k 准确率；框出所有并列最高值。
- interventions_heatmap：平均介入次数。
- length_ratio_heatmap：相对 x8 Student 的平均输出长度。

## 复现

从仓库根目录执行 `python3 figures/main_grid_x8_20260920/build_all.py`，然后执行 `python3 figures/main_grid_x8_20260920/build_gallery.py`。依赖 Python、NumPy、Matplotlib。每类图亦可单独运行对应 gen_*.py。

## 数据来源与校验

见 validation.json 的源文件路径与 SHA256。使用未四舍五入的数学 all_metrics.csv、代码逐题正确次数汇总、代码生成 token 总数和完整 x8 基线；未使用旧代码报告中的历史单次基线。统一长表为 plot_data.csv，8 × 27 = 216 行。代码指标由题级 histogram 和生成 token 总数重算；所有配置核验完整题数与 8N 输出数，以及 pass all 8 ≤ accuracy ≤ pass@8。

Teacher share = Teacher 输出 token 总和 / 全部输出 token 总和，不包括 prompt、丢弃草稿或 Teacher 打分开销，不是加速比。准确率为全部输出平均，pass@8 与 pass all 8 按题计算。均值包含生成超时；超时计错。

数学/推理的首轮与后续七轮在提取/评分签名、超时设置上不完全一致，部署并发也有变化。图注保留此限制，未声称严格受控实验或显著性。8 次 rollout 不是 8 倍独立题目。最高值只表示当前网格观察到的最大值。

## 排版

全部图无整体标题；数据集名称为面板标识。图注见 latex_includes.tex。2×4 总览适合大幅阅读；全部图另提供 *_print_part1/2，每张 2×2 面板、8.5 英寸宽（默认 LaTeX 片段使用拆分版），便于放入论文双栏并保持文字可读。热力图使用各面板独立色阶，禁止跨面板以深浅比较效果。
''')
print('Gallery, README and captions generated.')
