#!/bin/bash
#SBATCH --job-name=dm_max_verify
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_max_verify.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_max_verify.%j.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
python "${ROOT}/opd_scripts/verify_deepmath_maxlen_eval.py" \
  --new-root "${ROOT}/opd_eval_deepmath/maxlen_seed20260809" \
  --old-root "${ROOT}/opd_eval_deepmath/full_seed20260809"
