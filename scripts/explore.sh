export TASK=aime

export DATA_ROOT=data
export DATA_DIR=$DATA_ROOT/$TASK

python3 ../aime_1.py --local_dir $DATA_DIR

export N_GPUS=1
export BASE_MODEL=Qwen/Qwen2.5-Math-1.5B
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=ppo-$TASK-Qwen2.5-Math-1.5B-$JOB_NUM


python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=64\
    data.val_batch_size=1 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.use_dynamic_bsz=True \
    critic.optim.lr=1e-5 \
    critic.model.path=$BASE_MODEL \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.use_dynamic_bsz=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console','wandb'] \
    +trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=0 \
    trainer.test_freq=100000 \
    trainer.project_name=math \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=128000 2>&1 | tee verl_demo.log