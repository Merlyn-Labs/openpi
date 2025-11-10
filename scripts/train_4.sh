#!/bin/bash

# EXP_NAME="openpi_05_$(date +%Y%m%d_%H%M%S)"
EXP_NAME="openpi_05_20251110_180322"
echo "Experiment name: $EXP_NAME"

aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/pi05_single_task/openpi_05_20251109_221546/37000/ \
    /workspace/openpi/outputs/checkpoints/pi05_single_task/openpi_05_20251109_221546/37000/

CUDA_VISIBLE_DEVICES=6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMNIGIBSON_NO_SIGNALS=1 uv run scripts/train_val.py pi05_single_task_w_us \
    --exp_name="$EXP_NAME" \
    --resume \
    --batch_size=64 \
    --weight_loader.params_path=/workspace/openpi/outputs/checkpoints/pi05_single_task/openpi_05_20251109_221546/37000/params \
    --num_train_steps=50000 \
    --val_log_interval=3000
