#!/bin/bash

# NOTE: Update these paths to match your environment
cd "$(dirname "$0")/.."

# ═══════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════
MODEL_PATH="/path/to/olmocr-model"       # Base Qwen2.5-VL model (via olmOCR)
DATA_PATH="/path/to/olmocr-dataset"       # olmOCR training dataset

# ═══════════════════════════════════════════════════════════════
# CHECKPOINT RESUME (set to checkpoint dir to resume, empty for new training)
# ═══════════════════════════════════════════════════════════════
RESUME_FROM=""  

# ═══════════════════════════════════════════════════════════════
# DISTRIBUTED SETTINGS
# ═══════════════════════════════════════════════════════════════
NUM_GPUS=4
MASTER_PORT=29500

# ═══════════════════════════════════════════════════════════════
# OUTPUT DIRECTORY
# ═══════════════════════════════════════════════════════════════
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
CHECKPOINT_DIR="./outputs/diffuqwen-${TIMESTAMP}"
mkdir -p "$CHECKPOINT_DIR"

# ═══════════════════════════════════════════════════════════════
# NETWORK STABILITY
# ═══════════════════════════════════════════════════════════════
export NCCL_TIMEOUT=1800
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

# olmOCR uses: batch_size=1, grad_accum=32, lr=2e-5, max_token_len=8192, image_dim=1288
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train.py \
    --distributed \
    --model_path "$MODEL_PATH" \
    --dataset_root "$DATA_PATH" \
    --output_dir "$CHECKPOINT_DIR" \
    --max_length 8192 \
    --batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-4 \
    --weight_decay 0.05 \
    --max_steps -1 \
    --num_epochs 50 \
    --warmup_steps 1000 \
    --anneal_steps 10000 \
    --lora_r 32 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --save_steps 500 \
    --eval_steps 500 \
    --logging_steps 10 \
    --seed 42 \
    --bf16 \
    --resume_from_checkpoint "$RESUME_FROM" \
    2>&1 | tee -a "$CHECKPOINT_DIR/training.log"
