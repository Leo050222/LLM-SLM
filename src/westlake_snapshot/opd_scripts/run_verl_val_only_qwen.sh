#!/usr/bin/env bash
# Reproduce the formal run's validation path and four rollout-worker topology.

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 MODEL_PATH MODEL_NAME VALIDATION_OUTPUT_DIR [HYDRA_OVERRIDES...]" >&2
  exit 2
fi

MODEL_PATH=$1
MODEL_NAME=$2
VALIDATION_OUTPUT_DIR=$3
shift 3
DATA_DIR=/storage/yukaichengLab/wangxin/test_eiloeo/opd_data/gsm8k_math
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4097}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.45}

mkdir -p "${VALIDATION_OUTPUT_DIR}"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="['${DATA_DIR}/validation_512.parquet']" \
  data.val_files="['${DATA_DIR}/validation_512.parquet']" \
  data.train_batch_size=64 \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.shuffle=True \
  data.seed=20260806 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.seed=20260806 \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.use_v1=False \
  trainer.total_epochs=1 \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.validation_data_dir="${VALIDATION_OUTPUT_DIR}" \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.default_local_dir="${VALIDATION_OUTPUT_DIR}/unused_checkpoints" \
  trainer.resume_mode=disable \
  trainer.logger='["console"]' \
  trainer.project_name=qwen3_opd_eval \
  trainer.experiment_name="${MODEL_NAME}" \
  distillation.enabled=False \
  "$@"
