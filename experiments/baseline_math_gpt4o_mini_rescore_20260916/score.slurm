#!/bin/bash
#SBATCH --job-name=baseline_4omini
#SBATCH --partition=yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_rescore/baseline_math_gpt4o_mini_openlux_20260916/logs/job_%j.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_rescore/baseline_math_gpt4o_mini_openlux_20260916/logs/job_%j.err
set -Eeuo pipefail
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
OUT=${ROOT}/opd_rescore/baseline_math_gpt4o_mini_openlux_20260916
PY=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
# Authenticated relay has a fixed upstream; TLS still terminates at OpenLux.
PORT=$(${PY} -c 'import json; from pathlib import Path; print(json.loads(Path("/storage/yukaichengLab/wangxin/test_eiloeo/opd_rescore/baseline_math_gpt4o_mini_openlux_20260916/relay.json").read_text())["port"])')
trap 'touch "${OUT}/relay.stop"' EXIT
${PY} "${OUT}/score.py" --tunnel-port "${PORT}" --probe
${PY} "${OUT}/prepare.py"
${PY} "${OUT}/score.py" --tunnel-port "${PORT}" --workers 128

${PY} "${OUT}/compare.py"
