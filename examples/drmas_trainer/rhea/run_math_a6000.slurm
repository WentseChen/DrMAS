#!/bin/bash
#SBATCH --job-name=drmas-math
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Minimal Dr. MAS reproduction on Rhea (A6000, 48GB): Math task, 2 agents (Solver + Verifier).
# Downscaled from examples/drmas_trainer/run_math.sh (4 x H100, Qwen3-4B):
#   - default model Qwen3-1.7B with LLM sharing (both agents share one model)
#   - console logging only (set LOGGER="['console','wandb']" to enable wandb)
#
# Usage (from repo root):
#   mkdir -p logs && sbatch examples/drmas_trainer/rhea/run_math_a6000.slurm          # train
#   sbatch examples/drmas_trainer/rhea/run_math_a6000.slurm eval                       # eval only
#   MODEL=Qwen/Qwen3-4B MODEL_SHARING=False sbatch examples/drmas_trainer/rhea/run_math_a6000.slurm
# If you change --gres, NGPUS follows automatically from SLURM.
set -x

export DRMAS_CACHE=${DRMAS_CACHE:-$HOME/drmas_cache}
export HF_HOME=$DRMAS_CACHE/hf
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_$USER}   # keep short: ray socket paths have a length limit
mkdir -p "$HF_HOME" "$RAY_TMPDIR"

# module load cuda/12.4   # same as in setup_env.sh, if needed
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate DrMAS
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

NGPUS=${NGPUS:-${SLURM_GPUS_ON_NODE:-4}}
NCPUS=${SLURM_CPUS_PER_TASK:-32}
MODEL=${MODEL:-Qwen/Qwen3-1.7B}
MODEL_SHARING=${MODEL_SHARING:-True}
LOGGER=${LOGGER:-"['console']"}

MODE=${1:-train}
TRAIN_DATA="$HOME/data/drmas_math/train.parquet"
if [ "$MODE" == "eval" ]; then
    VAL_ONLY=True
    VAL_DATA="$HOME/data/drmas_math/test.parquet"
    train_data_size=32
    val_data_size=64
    val_group_size=16
else
    VAL_ONLY=False
    VAL_DATA="$HOME/data/drmas_math/test_sampled.parquet"
    train_data_size=32
    val_data_size=110
    val_group_size=1
fi

algorithm=grpo
group_size=8
group_by_agent_id=True # enable Dr. MAS

agent_ids='["Solver Agent","Verifier Agent"]'
model_ids="[\"$MODEL\",\"$MODEL\"]"
orchestra_type=math
max_loop_num=2

actor_optim_lr='[1e-6,1e-6]'
actor_ppo_micro_batch_size_per_gpu='[2,2]'

model_name_tag=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
experiment_name="drmas_rhea_share${MODEL_SHARING}_${model_name_tag}"

python3 -m verl.trainer.main_ppo \
    ray_init.num_cpus=$NCPUS \
    algorithm.adv_estimator=$algorithm \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    data.truncation='middle' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=null \
    actor_rollout_ref.actor.optim.lr=null \
    +agent.agent_specific_parameters.actor.optim.lr=$actor_optim_lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_adaptive_ppo_mini_batch_size=True \
    actor_rollout_ref.actor.ppo_mini_update_num=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null \
    +agent.agent_specific_parameters.actor.ppo_micro_batch_size_per_gpu=$actor_ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.group_by_agent_id=$group_by_agent_id \
    env.env_name=math \
    env.seed=0 \
    env.rollout.n=$group_size \
    env.rollout.val_n=$val_group_size \
    agent.agent_ids="$agent_ids" \
    agent.model_ids="$model_ids" \
    agent.model_sharing=$MODEL_SHARING \
    agent.orchestra_type=$orchestra_type \
    agent.orchestra.math.max_loop_num=$max_loop_num \
    trainer.critic_warmup=0 \
    trainer.logger="$LOGGER" \
    trainer.project_name='DrMAS_math' \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=2 \
    trainer.val_only=$VAL_ONLY \
    trainer.val_before_train=True
