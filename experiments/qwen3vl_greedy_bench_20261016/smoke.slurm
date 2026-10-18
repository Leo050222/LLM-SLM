#!/bin/bash
#SBATCH --job-name=qwen3vl_smoke
#SBATCH --partition=yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:H800:8
#SBATCH --exclusive
#SBATCH --mem=768G
#SBATCH --time=01:00:00
#SBATCH --exclude=gnho008,gnho009,gnho018
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_vlm/qwen3vl_instruct_greedy_20261016/logs/smoke_%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_vlm/qwen3vl_instruct_greedy_20261016/logs/smoke_%j.err
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1
exec /soft/slurm/bin/srun --kill-on-bad-exit=1 --cpu-bind=cores \
  /storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python -u \
  /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/qwen3vl_greedy_bench_20261016/node_runner.py --smoke
