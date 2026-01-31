#!/bin/bash

# Launch SGLang server for CriticLean or other LLM services
# Used for semantic consistency checking in ReForm

model_path=YOUR_MODEL_PATH  # e.g., /path/to/Criticlean-14B or /path/to/Qwen3-235B-A22B-Thinking

python -m sglang.launch_server \
    --model-path ${model_path} \
    --tp 4 \
    --trust-remote-code \
    --context-length 262144 \
    --max-running-requests 128 \
    --port 10086 \
    --host 0.0.0.0
