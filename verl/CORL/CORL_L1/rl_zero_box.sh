set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS
EXP_NAME='qwen2_5_vl_3b_corl_box_0.3'
export TORCH_SHOW_CPP_STACKTRACES=1

# Ray memory optimization settings
export RAY_memory_usage_threshold=0.85
export RAY_object_store_allow_slow_storage=1


OUTPUT_DIR="./checkpoints/${EXP_NAME}"

export WANDB_MODE=offline
export WANDB_DIR=${OUTPUT_DIR}/wandb
export WANDB_PROJECT=ICLR_COLR
export WANDB_NAME=${EXP_NAME}

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=./data/CORL-L1/level_1_new_prompt/train.parquet \
  data.val_files=./data/CORL-L1/level_1_new_prompt/test.parquet \
  data.train_batch_size=512 \
  data.max_prompt_length=4096 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.image_key=images \
  actor_rollout_ref.model.path=./models/Qwen2.5-VL-3B-Instruct \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=10 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=$ENGINE \
  +actor_rollout_ref.rollout.limit_images=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.n=5 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=10 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.save_freq=1000000 \
  trainer.test_freq=20 \
  trainer.logger=['console','wandb'] \
  trainer.val_before_train=false \
  trainer.project_name='ICLR_COLR' \
  trainer.experiment_name=${EXP_NAME} \
  trainer.default_local_dir=${OUTPUT_DIR} \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.total_epochs=10 $@
