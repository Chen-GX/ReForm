#!/bin/bash

# Convert HuggingFace checkpoint to Megatron-Core format
# This is required before running RL or SFT training with slime

cd ./slime
source ./scripts/models/qwen3-8B.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint ./Qwen3-8B \
    --save ./Qwen3-8B_torch_dist/
