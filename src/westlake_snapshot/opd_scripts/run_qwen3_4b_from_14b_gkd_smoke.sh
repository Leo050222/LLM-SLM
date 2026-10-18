#!/usr/bin/env bash
# Qwen3-4B <- Qwen3-14B on-policy distillation smoke test.
# Directly minimizes an approximation of KL(teacher || student) over teacher top-k tokens.

set -euo pipefail

STUDENT_MODEL=/storage/yukaichengLab/share/model/Qwen3-4B
TEACHER_MODEL=/storage/yukaichengLab/share/model/Qwen3-14B
DATA_DIR=/storage/yukaichengLab/wangxin/test_eiloeo/opd_data/gsm8k_smoke

# A single 8-GPU H800 node: 4 GPUs for the student hybrid engine, 2 for teacher vLLM.
TRAIN_GPUS=4
TEACHER_GPUS=2

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="['${DATA_DIR}/train.parquet']" \
  data.val_files="['${DATA_DIR}/test.parquet']" \
  data.train_max_samples=128 \
  data.val_max_samples=32 \
  data.train_batch_size=32 \
  data.max_prompt_length=1024 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${STUDENT_MODEL}" \
  actor_rollout_ref.model.use_remove_padding=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.max_model_len=2049 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
  trainer.n_gpus_per_node=${TRAIN_GPUS} \
  trainer.nnodes=1 \
  trainer.use_v1=False \
  trainer.total_epochs=1 \
  trainer.val_before_train=True \
  trainer.test_freq=4 \
  trainer.save_freq=-1 \
  trainer.logger='["console"]' \
  trainer.project_name=qwen3_opd_gkd \
  trainer.experiment_name=qwen3_4b_from_14b_gkd_smoke \
  distillation.enabled=True \
  distillation.n_gpus_per_node=${TEACHER_GPUS} \
  distillation.nnodes=1 \
  distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}" \
  distillation.teacher_models.teacher_model.inference.name=vllm \
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_GPUS} \
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.70 \
  distillation.teacher_models.teacher_model.inference.max_model_len=2049 \
  distillation.distillation_loss.loss_mode=forward_kl_topk \
  distillation.distillation_loss.topk=128 \
  distillation.distillation_loss.use_policy_gradient=False \
  distillation.distillation_loss.use_task_rewards=False \
  distillation.distillation_loss.loss_max_clamp=10.0 \
  distillation.distillation_loss.log_prob_min_clamp=-10.0
