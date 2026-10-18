# Qwen3-VL 模型下载记录

三个官方原始 BF16 模型已于 **2026-10-16 01:47:57（北京时间）** 下载、完成 CPU 校验并发布到 H800 共享存储。

最终根目录：

```text
/storage/yukaichengLab/share/model/
```

| 模型目录 | 文件总字节数 | 权重分片 | 固定 Hugging Face revision |
|---|---:|---:|---|
| `Qwen3-VL-2B-Instruct` | 4,266,648,961 | 1 | `89644892e4d85e24eaac8bacfd4f463576704203` |
| `Qwen3-VL-2B-Thinking` | 4,266,648,849 | 1 | `33e0ad94c327808ebc7d3bbc97e78e600f1eeecd` |
| `Qwen3-VL-8B-Thinking` | 17,545,915,778 | 4 | `92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b` |

合计 **26,079,213,588 字节（26.08 GB）**，40 个官方文件、6 个权重分片。大小包含模型配置、分词器和处理器文件，不包含下载缓存或本次生成的校验记录。

## 校验范围

- 全部 40 个计划文件的大小核对及 SHA-256 计算；6 个权重分片的 SHA-256 均与固定官方 revision 的预期值一致。
- Safetensors 文件头读取成功；8B 的索引与 4 个权重分片逐项一致。两个 2B 模型各 625 个 tensor，8B 模型共 750 个 tensor。
- 配置、分词器和图像处理器均可完全离线读取。CPU 图文处理检查通过，图像 token 数与图像网格一致。
- 全局状态为 `COMPLETE`，校验退出码为 `0`；三个最终模型目录均有 `VERIFIED.json`。

最终模型路径均为真实目录，下载缓存未放入最终目录。共享文件系统的发布方式与恢复信息见各模型的 `PUBLICATION.json`；暂存源保留用于恢复，权重使用硬链接，不重复占用一份权重空间。

本次仅完成模型下载与 CPU 校验，未运行 GPU 推理、benchmark 或评分 API。

## 报告

- [全局完成报告](reports/COMPLETE.json)
- [全局校验报告](reports/VERIFIED.json)
- [下载状态与最终路由](reports/status.json)
- [下载及全局校验日志](reports/run.log)
- [两个 2B 模型提前发布日志](reports/verify_two_ready.log)
- [本地报告文件 SHA-256 清单](reports/REPORTS_MANIFEST.json)
- 各模型的 `SOURCE.json`、`VERIFIED.json`、`PUBLICATION.json` 保存在 `reports/models/<模型名>/`。

`publish_ready.log` 保留首次发布因共享文件系统不支持原子 no-replace rename 而失败的历史诊断；后续已使用安全的目标目录预留方式成功发布，最终以 `COMPLETE.json` 和模型内的 `VERIFIED.json` 为准。

远端任务及报告目录：

```text
/storage/yukaichengLab/wangxin/test_eiloeo/opd_data/model_downloads/qwen3vl_20261016/
```
