#!/bin/bash
################## PATHS ##################
cd "$(dirname "$0")/../../../.."

# Base Model & Vision Encoder
LLADA_8B_INSTRUCT="/path/to/lavida-ckpts/lavida-llada-hd"
VISION_MODEL_VERSION="/path/to/lavida-ckpts/google-siglip-so400m-patch14-384"

DATA_PATH="/path/to/lavida-dataset/data/olmocr/olmocr_train_final.json"
IMG_PATH="/path/to/lavida-dataset"

OUTPUT_ROOT="/path/to/output-checkpoints"

################## CONFIG ##################

MID_RUN_NAME="lavida-stage2-olmocr-qwen2vlmerger"
echo "Starting Run: ${MID_RUN_NAME}"

# Hardware Setup
NUM_GPUS=4
PORT=29500  

export ALWASY_DO_2DPOOL=0
export NOT_ALWASY_DO_2DPOOL=1

# Debug/Fix flags
export SELECT_ONE_INDEX=1
export DEBUG_FIX_PADDIN=1

################## TRAIN ##################

torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path "${LLADA_8B_INSTRUCT}" \
    --version "llada" \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMG_PATH}" \
    --mm_tunable_parts "mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr 2e-6 \
    --vision_tower "${VISION_MODEL_VERSION}" \
    --mm_projector_type "qwen2_vl_merger" \
    --image_aspect_ratio "anyres" \
    --image_grid_pinpoints "[(384, 768), (768, 384), (768, 768), (1152, 384), (384, 1152), (1152, 1152), (768, 1152), (1152, 768)]" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --mm_patch_merge_type "flat" \
    --bf16 True \
    --run_name "${MID_RUN_NAME}" \
    --output_dir "${OUTPUT_ROOT}/${MID_RUN_NAME}" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 250 \
    --save_total_limit 2 \
    --learning_rate 2e-7 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to "none" \
    --dataloader_drop_last True \
    --attn_implementation "sdpa" \
    --resume_from_checkpoint "latest" \
    --lmms_eval_generate_tasks "" \
    --lr_scheduler_kwargs '{"min_lr_rate":0.1}' \
    --pretrain_mm_mlp_adapter "/path/to/output-checkpoints/lavida-stage2-olmocr-qwen2vlmerger-projector-only-finetune/checkpoint-500/mm_projector.bin" 
