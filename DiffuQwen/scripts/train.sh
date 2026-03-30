#!/bin/bash

# NOTE: Update these paths to match your environment
cd "$(dirname "$0")/.."

# ═══════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════
MODEL_PATH="/path/to/olmocr-model"       # Base Qwen2.5-VL model (via olmOCR)
DATA_PATH="/path/to/olmocr-dataset"       # olmOCR training dataset
OUTPUT_DIR="./checkpoints/diffuqwen-hf-$(date +%Y%m%d-%H%M%S)"

# Set this to resume from a HF Trainer checkpoint (leave empty for fresh start)
CHECKPOINT_PATH=""

mkdir -p "$OUTPUT_DIR"

# ═══════════════════════════════════════════════════════════════
# DISTRIBUTED SETTINGS
# ═══════════════════════════════════════════════════════════════
NUM_GPUS=4

export NCCL_TIMEOUT=1800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

# Build resume arg only if checkpoint path is set
RESUME_ARG=""
if [ -n "$CHECKPOINT_PATH" ]; then
    RESUME_ARG="--resume_from_checkpoint $CHECKPOINT_PATH"
fi

torchrun \
    --nproc_per_node=$NUM_GPUS \
    train.py \
    --model_path "$MODEL_PATH" \
    --dataset_root "$DATA_PATH" \
    $RESUME_ARG \
    --max_length 8192 \
    --lora_r 32 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --anneal_steps 10000 \
    --output_dir "$OUTPUT_DIR" \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-4 \
    --weight_decay 0.05 \
    --warmup_steps 1000 \
    --num_train_epochs 50 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 500 \
    --save_steps 250 \
    --save_total_limit 20 \
    --bf16 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory \
    --remove_unused_columns false \
    --ddp_find_unused_parameters false \
    --report_to none \
    --seed 42 \
    2>&1 | tee "$OUTPUT_DIR/training.log"
