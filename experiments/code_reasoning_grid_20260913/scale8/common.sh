export PYTHONDONTWRITEBYTECODE=1
export OPD_CODE_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_online_code/20260911_chunk64_k8_2nodes
export GRID_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_code/code_reasoning_grid_20260913
export OPD_ONLINE_RUN=$GRID_ROOT/slots/job_$SLURM_JOB_ID
export OPD_PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
export SCALE_CODE=/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/code_reasoning_grid_scale8_20260913
mkdir -p "$OPD_ONLINE_RUN/logs"
if [[ -f "$GRID_ROOT/ALL_GENERATION_COMPLETE.json" ]]; then exit 0; fi
