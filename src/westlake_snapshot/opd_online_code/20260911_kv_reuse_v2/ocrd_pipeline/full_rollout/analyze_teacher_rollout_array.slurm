#!/usr/bin/env bash
#SBATCH --job-name=score_tfull_n8
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/teacher_full_qwen3_14b_n8_seed20260826/logs/%j.deprecated_array.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/teacher_full_qwen3_14b_n8_seed20260826/logs/%j.deprecated_array.err

set -euo pipefail
echo "DEPRECATED: do not submit CPU arrays; use analyze_teacher_rollout_single_job.slurm" >&2
exit 2
