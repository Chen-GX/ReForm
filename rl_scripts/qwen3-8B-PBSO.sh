#!/bin/bash

# PBSO (Prospective Bounded Sequence Optimization) Training Script for Qwen3-8B
# This script runs RL training with multi-step reward and discounted propagation

# Clean up any existing processes
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

pip install json5

set -x

export NCCL_SOCKET_IFNAME=eth0
export NCCL_TIMEOUT=60000000
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_NET_PLUGIN=none

export NPROC_PER_NODE=${1:-${MLP_WORKER_GPU:-${KUBERNETES_CONTAINER_RESOURCE_GPU:-8}}} 
export WORLD_SIZE=${2:-${MLP_WORKER_NUM:-${WORLD_SIZE:-1}}}
export RANK=${3:-${MLP_WORKER_RACK_RANK_INDEX:-${MLP_ROLE_INDEX:-${RANK:-0}}}}
export MASTER_ADDR=${4:-${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}}

# Environment variables for ReForm
export STEP_REWARD_TOKEN_ID=151670  # Token ID for step boundary markers
export CRITICLEAN_URL="http://127.0.0.1:10086/v1"  # CriticLean server address

timestamp=$(date +%Y%m%d_%H%M%S)

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "./slime/scripts/models/qwen3-8B.sh"

# PBSO reward shaping configuration
reward_shape="discounted_with_clip"
reward_shape_gamma=0.5
SAVE_PATH=YOUR_SAVE_PATH

CKPT_ARGS=(
   --hf-checkpoint YOUR_SFT_MODEL_PATH
   --ref-load YOUR_SFT_MODEL_PATH_FOR_REF
   --load ${SAVE_PATH}
   --save ${SAVE_PATH}
   --save-interval 5
)

ROLLOUT_ARGS=(
   --prompt-data ./data/rl_data/rl_train.jsonl
   --input-key question
   --label-key label
   --rollout-shuffle
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 16
   --rollout-max-response-len 20480
   --rollout-temperature 1.0
   --rollout-top-p 0.95

   --global-batch-size 512
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 5
   --eval-prompt-data TEST_DATA_NAME TEST_DATA_PATH
   --n-samples-per-eval-prompt 3
   --eval-max-response-len 20480
   --eval-temperature 0.6
   --eval-top-p 0.95
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-lean
   --wandb-group grpo-${reward_shape}_${reward_shape_gamma}
   --wandb-key ${WANDB_API_KEY}
)

SGLANG_ARGS=(
   --sglang-server-concurrency 64
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ReForm custom arguments for PBSO
CUSTOM_ARGS=(
   --rollout-function-path lean_plugins.generate.lean_generate_rollout
   --custom-rm-path lean_plugins.reward.reward_func
   --custom-reward-post-process-path lean_plugins.adv_utils.custom_reward_post_process_func
   --custom-adv-returns-function-path lean_plugins.adv_utils.compute_step_advantages_and_returns

   --distributed-timeout-minutes 60

   --use-tis

   --reward-shaping ${reward_shape}
   --reward-shaping-gamma ${reward_shape_gamma}
)

if [ $RANK -eq 0 ]; then
    mkdir -p ${SAVE_PATH}
    export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
    export no_proxy="127.0.0.1,${MASTER_ADDR}"
    ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NPROC_PER_NODE} --port=6379 --disable-usage-stats

    RUNTIME_ENV_JSON="{
    \"env_vars\": {
        \"PYTHONPATH\": \"/root/Megatron-LM/:/YOURPATH/ReForm/\",
        \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
        \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
         \"no_proxy\": \"${no_proxy}\",
         \"MASTER_ADDR\": \"${MASTER_ADDR}\",
         \"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME}\"
    }
    }"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ./slime/train.py \
   --actor-num-nodes ${WORLD_SIZE} \
   --actor-num-gpus-per-node ${NPROC_PER_NODE} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]} 2>&1 | tee ${SAVE_PATH}/log_${timestamp}.log

else
   sleep 60
   pkill -9 sglang ; ray stop --force ; pkill -9 python ; ray start --block --address=${MASTER_ADDR}:6379 --num-gpus ${NPROC_PER_NODE} --disable-usage-stats
fi
