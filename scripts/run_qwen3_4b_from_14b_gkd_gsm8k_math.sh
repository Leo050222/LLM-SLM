#!/usr/bin/env bash
# Full one-epoch OPD run: Qwen3-4B student <- Qwen3-14B teacher.
# Loss: top-k approximation to forward KL, KL(teacher || student), direct backpropagation.

set -euo pipefail

STUDENT_MODEL=/storage/yukaichengLab/share/model/Qwen3-4B
TEACHER_MODEL=/storage/yukaichengLab/share/model/Qwen3-14B
DATA_DIR=/storage/yukaichengLab/wangxin/test_eiloeo/opd_data/gsm8k_math
CHECKPOINT_DIR=/storage/yukaichengLab/wangxin/test_eiloeo/opd_checkpoints/qwen3_4b_from_14b_gkd_gsm8k_math

# Four H800s host the student FSDP/rollout workers; two H800s host the teacher TP=2 engine.
TRAIN_GPUS=4
TEACHER_GPUS=2
TRAIN_BATCH_SIZE=64
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=2048
MAX_MODEL_LEN=4097

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="['${DATA_DIR}/train.parquet']" \
  data.val_files="['${DATA_DIR}/validation_512.parquet']" \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.max_prompt_length=${MAX_PROMPT_LENGTH} \
  data.max_response_length=${MAX_RESPONSE_LENGTH} \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.shuffle=True \
  data.seed=20260806 \
  actor_rollout_ref.model.path="${STUDENT_MODEL}" \
  actor_rollout_ref.model.use_remove_padding=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra','hf_model']" \
  actor_rollout_ref.actor.checkpoint.load_contents="['model','optimizer','extra']" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.seed=20260806 \
  actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
  trainer.n_gpus_per_node=${TRAIN_GPUS} \
  trainer.nnodes=1 \
  trainer.use_v1=False \
  trainer.total_epochs=1 \
  trainer.val_before_train=True \
  trainer.test_freq=234 \
  trainer.save_freq=50 \
  trainer.default_local_dir="${CHECKPOINT_DIR}" \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.resume_mode=auto \
  trainer.logger='["console"]' \
  trainer.project_name=qwen3_opd_gkd \
  trainer.experiment_name=qwen3_4b_from_14b_gkd_gsm8k_math \
  distillation.enabled=True \
  distillation.n_gpus_per_node=${TEACHER_GPUS} \
  distillation.nnodes=1 \
  distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}" \
  distillation.teacher_models.teacher_model.inference.name=vllm \
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_GPUS} \
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.70 \
  distillation.teacher_models.teacher_model.inference.max_model_len=${MAX_MODEL_LEN} \
  distillation.distillation_loss.loss_mode=forward_kl_topk \
  distillation.distillation_loss.topk=128 \
  distillation.distillation_loss.use_policy_gradient=False \
  distillation.distillation_loss.use_task_rewards=False \
  distillation.distillation_loss.loss_max_clamp=10.0 \
  distillation.distillation_loss.log_prob_min_clamp=-10.0 \
  "$@"
