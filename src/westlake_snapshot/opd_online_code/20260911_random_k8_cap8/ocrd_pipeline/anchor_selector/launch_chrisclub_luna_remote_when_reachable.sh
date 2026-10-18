#!/bin/bash
set -u

LOCAL_ROOT=/Users/leolee/Documents/OPD
LOCAL_MODULE=${LOCAL_ROOT}/ocrd_pipeline/anchor_selector
STATE_DIR=${LOCAL_ROOT}/tmp/chrisclub_luna_remote_launch_20260824
REMOTE_HOST=wangxin@172.16.78.35
REMOTE_MODULE=/storage/yukaichengLab/wangxin/test_eiloeo/ocrd_pipeline/anchor_selector
REMOTE_PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python

mkdir -p "${STATE_DIR}"
attempt=0
while true; do
  attempt=$((attempt + 1))
  chosen_port=
  for port in 10024 10025; do
    if rsync -az \
      -e "ssh -p ${port} -o ConnectTimeout=20 -o ServerAliveInterval=10 -o ServerAliveCountMax=2" \
      "${LOCAL_MODULE}/run_anchor_judge_batch.py" \
      "${LOCAL_MODULE}/openlux_client.py" \
      "${LOCAL_MODULE}/config_gpt56_luna.json" \
      "${LOCAL_MODULE}/submit_deepmath_train25_anchor_judge_chrisclub_luna.slurm" \
      "${LOCAL_MODULE}/.secrets.env" \
      "${REMOTE_HOST}:${REMOTE_MODULE}/"; then
      chosen_port=${port}
      break
    fi
  done

  if [[ -n "${chosen_port}" ]]; then
    submission=$(ssh -p "${chosen_port}" \
      -o ConnectTimeout=30 \
      -o ServerAliveInterval=10 \
      -o ServerAliveCountMax=3 \
      "${REMOTE_HOST}" \
      "bash -ilc 'set -euo pipefail; chmod 600 ${REMOTE_MODULE}/.secrets.env; source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh; conda activate verl-opd; cd /storage/yukaichengLab/wangxin/test_eiloeo/ocrd_pipeline; python -m unittest discover -s anchor_selector/tests -v; set -a; source ${REMOTE_MODULE}/.secrets.env; set +a; unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; curl -fsS --max-time 30 -H \"Authorization: Bearer \$OPENLUX_API_KEY\" http://38.145.220.6:9000/v1/models >/dev/null; mkdir -p /storage/yukaichengLab/wangxin/test_eiloeo/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/judge_remote_gpt56_luna_batch4/logs; /soft/slurm/bin/sbatch --parsable ${REMOTE_MODULE}/submit_deepmath_train25_anchor_judge_chrisclub_luna.slurm'" 2>&1)
    status=$?
    printf '%s\n' "${submission}" >> "${STATE_DIR}/submission.log"
    if [[ ${status} -eq 0 ]]; then
      job_id=$(printf '%s\n' "${submission}" | tail -n 1 | tr -cd '0-9')
      if [[ -n "${job_id}" ]]; then
        printf '%s\n' "${job_id}" > "${STATE_DIR}/job_id"
        printf '%s\n' "${chosen_port}" > "${STATE_DIR}/ssh_port"
        touch "${STATE_DIR}/SUBMITTED"
        exit 0
      fi
    fi
  fi

  printf 'attempt=%d time=%s cluster_unreachable_or_submit_failed\n' \
    "${attempt}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${STATE_DIR}/retry.log"
  sleep 30
done
