#!/bin/bash
#SBATCH --job-name=dm_17miss14_vis
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_17miss14_vis.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_17miss14_vis.%j.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
RUN=${ROOT}/opd_eval_deepmath/maxlen_seed20260809
python ${ROOT}/opd_scripts/build_deepmath_17b_miss14b_token_vis.py \
  --root ${RUN} \
  --template ${ROOT}/opd_scripts/deepmath_17b_miss14b_fragment.html \
  --output ${RUN}/deepmath_17b_miss14b_token_difference.html
