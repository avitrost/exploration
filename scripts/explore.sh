export TASK=math

export DATA_ROOT=data
export DATA_DIR=$DATA_ROOT/$TASK

python3 examples/data_preprocess/preprocess_small_math.py --local_dir $DATA_DIR

export N_GPUS=2
export BASE_MODEL=Qwen/Qwen2.5-3B
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=ppo-$TASK-qwen2.5-3B-$JOB_NUM


python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=50 \
    data.val_batch_size=50 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=50 \
    actor_rollout_ref.actor.ppo_micro_batch_size=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
    critic.optim.lr=1e-5 \
    critic.model.path=$BASE_MODEL \
    critic.ppo_micro_batch_size=2 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console','wandb'] \
    +trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=0 \
    trainer.test_freq=10000 \
    trainer.project_name=math \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=128 2>&1 | tee verl_demo.log