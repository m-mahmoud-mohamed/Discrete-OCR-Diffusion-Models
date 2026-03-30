#!/bin/bash
# Parallel multi-GPU inference for LaViDa-OCR
# Usage: bash run_parallel.sh

LOG_DIR="./logs"
mkdir -p $LOG_DIR
NUM_GPUS=4
SCRIPT_PATH="./predict_parallel.py"

echo "Starting parallel inference on ${NUM_GPUS} GPUs..."

for ((i=0; i<NUM_GPUS; i++)); do
    echo "Launching Worker $i on GPU $i..."
    
    # Isolate each worker to a single GPU
    export CUDA_VISIBLE_DEVICES=$i
    
    python -u $SCRIPT_PATH \
        --chunk-idx $i \
        --num-chunks $NUM_GPUS \
        > "${LOG_DIR}/log_worker_$i.txt" 2>&1 &

    unset CUDA_VISIBLE_DEVICES
done

wait
echo "Done. Check ${LOG_DIR}/ for per-worker logs."
