# H800 大规模集群使用指南

## 基本约定

- 集群采用 HPC 架构：登录节点（login）不能使用 GPU；GPU 在计算节点上使用。
- 计算节点不联网。环境配置、下载数据集和模型权重应在登录节点完成；需要依赖特定 CUDA 版本的包（如 `flash-attn`）可先在登录节点下载 wheel 或源码，再到 GPU 节点离线安装。
- 大模型集群没有预配置 `module` 命令。作业脚本中应按实际需要加载兼容的 CUDA 环境；当前最高支持 CUDA 12.5，通常可向下兼容。
- 每个计算节点配备 8 张 H800 GPU。
- 长时间训练或推理必须通过 Slurm 提交专用作业；禁止在交互环境或开发机中运行长时间训练。

## 登录与共享资源

使用分配给你的用户名登录：

```bash
ssh wangxin@172.16.78.35 -p 10024
```

端口也可能为 `10025`，以管理员提供的信息为准。不要在仓库文件、脚本或日志中记录密码；没有账号请联系集群管理员。

公共数据集与开源模型权重统一存放在以下目录。优先复用，禁止重复下载造成磁盘浪费：

```text
/storage/yukaichengLab/share/datasets
/storage/yukaichengLab/share/model
```

个人工作目录固定为：

```text
/storage/yukaichengLab/wangxin/test_eiloeo
```

项目代码、作业脚本、日志和个人生成的数据均应保存在该目录下；不要直接修改共享数据集或模型目录。

### 输出路径约束（强制）

- 以后所有实验输出（checkpoint、rollout、评测结果、Hydra run 目录、Slurm stdout/stderr、可视化和临时产物）必须写入 `/storage/yukaichengLab/wangxin/test_eiloeo/<分类目录>`。
- 禁止把新的输出直接写入 `/storage/yukaichengLab/wangxin/outputs` 或用户根目录的其他新目录；Hydra 任务必须显式设置 `hydra.run.dir` 到 `test_eiloeo` 下。
- 共享模型和数据只能读取；个人 conda 环境仍可保留在 `/storage/yukaichengLab/wangxin/miniconda3`。

## 开发机：仅用于短时调试

开发机提交脚本如下，会独占一个节点的 8 张 GPU，并在 **3 小时后自动结束**：

```bash
sbatch /storage/yukaichengLab/share/dev/scripts/apply_dev.sh
```

- 不得修改脚本中的 3 小时时限。
- 旧的 `inf_task.sh` / `infinite_run` 流程已弃用，不要提交或依赖它。
- 开发机适合环境验证、短时调试；验证完成后退出并提交正式 Slurm 作业。

## 标准工作流

### 1. 登录节点初始化与资源查询

```bash
source ~/.bashrc
sinfo
```

`sinfo` 用于查看可用分区和计算节点。提交前确认目标节点确实空闲。

### 2. 在计算节点进行交互式调试

进入指定计算节点后，加载 shell 配置并进行短时测试：

```bash
ssh gnho030
source ~/.bashrc
conda activate <环境名>
nvidia-smi
```

退出该会话会结束交互式任务并释放资源：

```bash
exit
```

如需限定 GPU，避免多个任务误用同一张卡，可使用：

```bash
ssh -t gnho030 'CUDA_VISIBLE_DEVICES=<device_num> bash -l'
```

### 3. 提交正式 Slurm 作业

在登录节点创建并提交作业脚本。例如：

```bash
#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:<GPU数量>
#SBATCH --nodelist=gnho030
#SBATCH -J qwen_grpo
#SBATCH -o qwen_grpo.%j.log
#SBATCH -e qwen_grpo.%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<邮箱地址>

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

source /storage/yukaichengLab/wangxin/anaconda3/etc/profile.d/conda.sh
conda activate <环境名>
cd /storage/yukaichengLab/wangxin/test_eiloeo

bash <训练或推理脚本>.sh

echo "End time: $(date)"
conda deactivate
```

提交命令：

```bash
sbatch <作业脚本>.bash
```

提交成功后会返回作业 ID。

## 作业监控与管理

```bash
# 查看自己的作业及节点分配
squeue -u <用户名>

# 持续查看标准输出
tail -f qwen_grpo.<作业ID>.log

# 查看错误或已有输出
cat qwen_grpo.<作业ID>.err
cat qwen_grpo.<作业ID>.log

# 查看作业与节点详情
scontrol show job <作业ID>

# 取消作业
scancel <作业ID>
```

也可以 SSH 到作业所在计算节点并使用 `nvidia-smi` 查看 GPU 实际占用。登录节点提交正式任务前，务必退出不再使用的交互调试环境，释放其占用的资源。

## 文件传输

可使用 `scp` 或 `rsync`。例如从本地上传：

```bash
rsync -av --progress -e "ssh -p 10025" <本地来源路径> \
  wangxin@172.16.78.35:/storage/yukaichengLab/wangxin/test_eiloeo/<目标目录>
```
