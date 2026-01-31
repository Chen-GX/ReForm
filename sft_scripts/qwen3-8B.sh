#!/bin/bash

# SFT (Supervised Fine-Tuning) Script for Qwen3-8B
# This script runs supervised fine-tuning before PBSO RL training

# Clean up any existing processes
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -x

export NPROC_PER_NODE=${1:-${MLP_WORKER_GPU:-${KUBERNETES_CONTAINER_RESOURCE_GPU:-8}}} 
export WORLD_SIZE=${2:-${MLP_WORKER_NUM:-${WORLD_SIZE:-1}}}
export RANK=${3:-${MLP_WORKER_RACK_RANK_INDEX:-${MLP_ROLE_INDEX:-${RANK:-0}}}}
export MASTER_ADDR=${4:-${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}}

timestamp=$(date +%Y%m%d_%H%M%S)

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "./slime/scripts/models/qwen3-8B.sh"

SAVE_PATH=YOUR_SAVE_PATH_${timestamp}
WANDB_NAME=${timestamp}

CKPT_ARGS=(
   --hf-checkpoint YOUR_MODEL_PATH
   --ref-load YOUR_REF_LOAD_PATH
   --load ${SAVE_PATH}
   --save ${SAVE_PATH}
   --save-interval 2000
)

SFT_ARGS=(
   --prompt-data ./data/sft_data/sft_data.jsonl
   --input-key messages
   --rollout-shuffle
   --num-epoch 3
   --rollout-batch-size 512
   --global-batch-size 512

   --loss-type sft_loss
   --loss-mask-type qwen3
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

PERF_ARGS=(
   --tensor-model-parallel-size 8
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.03
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project YOUR_PROJECT_NAME
   --wandb-group ${WANDB_NAME}
   --wandb-key ${WANDB_API_KEY}
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

if [ $RANK -eq 0 ]; then
   mkdir -p ${SAVE_PATH}
    export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
    export no_proxy="127.0.0.1,${MASTER_ADDR}"
    ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NPROC_PER_NODE} --port=6379 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

    RUNTIME_ENV_JSON="{
    \"env_vars\": {
        \"PYTHONPATH\": \"/root/Megatron-LM/\",
        \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
        \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
         \"no_proxy\": \"${no_proxy}\",
         \"MASTER_ADDR\": \"${MASTER_ADDR}\"
    }
    }"

    ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 ./slime/train_async.py \
    --actor-num-nodes ${WORLD_SIZE} \
    --actor-num-gpus-per-node ${NPROC_PER_NODE} \
    ${MODEL_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${SFT_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${WANDB_ARGS[@]} \
    ${PERF_ARGS[@]} \
    ${EVAL_ARGS[@]} \
    ${MISC_ARGS[@]} 2>&1 | tee ${SAVE_PATH}/log_${timestamp}.log

else
   sleep 60
   pkill -9 sglang ; ray stop --force ; pkill -9 python ; ray start --block --address=${MASTER_ADDR}:6379 --num-gpus ${NPROC_PER_NODE} --disable-usage-stats
fi
