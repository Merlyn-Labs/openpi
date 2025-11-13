#!/bin/bash

EXP_NAME="openpi_05_$(date +%Y%m%d_%H%M%S)"
echo "Experiment name: $EXP_NAME"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMNIGIBSON_NO_SIGNALS=1 uv run scripts/train_val.py pi05_b1k_22_TASKS_oversample \
    --exp_name="$EXP_NAME" \
    --overwrite \
    --batch_size=112 \
    --weight_loader.params_path=gs://openpi-assets/checkpoints/pi05_base/params \
    --num_train_steps=100000 \
    --val_log_interval=3000
