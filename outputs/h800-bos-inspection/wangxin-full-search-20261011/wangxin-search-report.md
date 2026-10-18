# wangxin 搜索结果

本次检查当前凭据可见的 16 个 BOS 桶，已完成 16 个。
按对象名称/路径检索 wangxin，不读取文件内容。累计检查 60,850,654 条对象元信息，去重后字符串命中 1,072,510 个对象。
大小按对象元信息相加，GB/TB 为十进制；不同路径下内容相同的副本没有按内容去重。

| 明确命名为 wangxin 的目录 | 对象数 | 总大小（GB） |
|---|---:|---:|
| `bos://auwomo-data-processed/wangxin/` | 929,486 | 615.175 |
| `bos://auwomo-data/auwomo-data-processed/wangxin/` | 142,644 | 105.701 |

`auwomo-data-processed/wangxin/` 内有 `simforge/`、`libero_plus/`、`simforge-backups/` 和 `simforge-migration/`。另一个桶的 `auwomo-data-processed/wangxin/` 内有 simforge、备份及迁移包。
`bos://auwomo-data/wangxin_4090/` 只有零字节目录标记。

另发现相关旧实验目录 `bos://auwomo-data/wanxin_westlake/`（wanxin 少一个 g）：完整前缀检查已完成，共 4,018,863 个对象，7.365 TB。
这里有 GenMol、LLaDA、MMaDA、REditor、RGroundBench、nli、模型检查点、运行环境和实验输出；其中的文件路径包含旧 `/storage/yukaichengLab/wangxin/GenMol/` 引用，以及 `pytest-of-wangxin` 临时目录。目录名和路径关联很强，但本次未核实所有文件的归属。

GenMol `model_v2.ckpt` 和 LLaDA 一个模型分片的只读元信息查询成功；未下载权重内容或验证完整性。
已定位的 wangxin 命中路径及完整 wanxin_westlake 前缀中，未发现 eiloeo 名称；压缩包内部未检索。
当前 H800 PVC 的完整目录遍历（无深度限制）成功完成，wangxin 路径命中 0 个。BOS 中这些对象尚未在当前 /data 下找到对应目录。

完整对象清单按桶保存在 `auwomo-data-wangxin-matches.csv`、`auwomo-data-processed-wangxin-matches.csv`；逐桶扫描状态是 `summary.json`。
相关旧实验目录的完整名称检查记录是 `westlake-alias-full-scan.json`；权重元信息核对结果是 `westlake-alias-weight-head-check.json`。
