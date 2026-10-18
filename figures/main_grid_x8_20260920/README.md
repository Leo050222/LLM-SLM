# 主表可视化（8 次 rollout）

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
