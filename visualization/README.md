# 可视化入口

本目录通过相对符号链接集中展示浏览器、复算脚本、报告和图表；原物理路径保留，避免改变分析代码的路径计算。结果与实验配置的对应关系见 [实验总表](../docs/experiments/ALL_EXPERIMENTS.md)，更多代码说明见 [CODE_MAP.md](../docs/project/CODE_MAP.md)。

| 分类入口 | 实际目录 | 用途 |
| --- | --- | --- |
| `browser` | 根目录 `code/bridge-rollout-browser/` | direct 与 after-bridge 回答对比浏览器 |
| `analysis` | 根目录 `analysis/` | bridge、长度、准确率和分布分析复算 |
| `reports` | 根目录 `reports/` | 离线 HTML、结果报告及报告生成代码 |
| `figures` | 根目录 `figures/` | 图表与相应源数据 |

## Bridge 对比浏览器

有效入口为 `index.html` → `src/main.tsx` → `app/page.tsx` 与 `app/globals.css`。项目使用 React + Vite，需要 Node ≥ 22.13.0 及 `public/data/`。

```bash
# 在仓库根执行
cd code/bridge-rollout-browser
pnpm install --frozen-lockfile
pnpm dev
```

开发地址为 `http://127.0.0.1:4173`；`pnpm build` 生成构建产物，`pnpm preview` 预览构建。依赖和构建目录可重建，`public/data/` 是输入数据。

数据导出代码为 [export_bridge_100pp_browser_data.py](../scripts/export_bridge_100pp_browser_data.py)，读取西湖的原始 trace 与 x8 scores，向 stdout 输出 gzip tar。该脚本固定筛选 strict direct 0/8 → after-bridge 8/8 的历史样本，应先核对脚本中的远端数据路径和选样约束，再生成数据包。

## 离线 token heatmap 与报告

在仓库根执行：

```bash
python scripts/export_opd_token_heatmap.py /path/to/diagnostic/tokens.parquet \
  --output reports/example_token_heatmap.html --title "Token KL diagnostic"
```

导出器为 [export_opd_token_heatmap.py](../scripts/export_opd_token_heatmap.py)，模板与字段说明见 [opd_token_heatmap README](../tools/opd_token_heatmap/README.md)。生成的 HTML 可直接离线打开。

其他入口包括 [baseline gap 页面](../scripts/build_deepmath_baseline_gap_vis.py)、[student-top16 trace 页面](../scripts/build_deepmath_student_top16_trace_vis.py) 和 [1.7B 对比 14B token 页面](../scripts/build_deepmath_17b_miss14b_token_vis.py)。先定位对应诊断与评分产物，再运行导出器；页面本身不替代原始分数。

## 图表复算约定

`analysis/plot_*.py` 和 `reports/**/plot_*.py` 读取相邻 CSV / JSON 或脚本声明的数据路径。保留源指标字段、单位、样本数和采样口径，明确区分平均准确率、pass@8、all@8 与长度统计。新图应同时记录输入文件和生成命令；输出到西湖项目根目录下的分类目录，或本地明确指定的 `reports/` / `figures/`。

历史报告中包含凭据材料的文件不进入发布快照。生成新页面时不把 API 配置、请求授权头或本地 secret env 嵌入 HTML / JSON。
