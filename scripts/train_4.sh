#!/bin/bash

# IMPORTANT NOTE: STILL NEED TO ACTUALLY IMPLEMENT THIS CONFIG AND THE OVERSAMPLING LOGIC

EXP_NAME="openpi_05_$(date +%Y%m%d_%H%M%S)"
echo "Experiment name: $EXP_NAME"

CUDA_VISIBLE_DEVICES=6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMNIGIBSON_NO_SIGNALS=1 uv run scripts/train_val.py pi05_single_task_focus_on_boundaries \
    --exp_name="$EXP_NAME" \
    --resume \
    --batch_size=64 \
    --weight_loader.params_path=gs://openpi-assets/checkpoints/pi05_base/params \
    --num_train_steps=50000 \
    --val_log_interval=3000
