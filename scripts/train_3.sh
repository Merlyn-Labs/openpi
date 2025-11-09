#!/bin/bash

EXP_NAME="openpi_0_$(date +%Y%m%d_%H%M%S)"
echo "Experiment name: $EXP_NAME"

CUDA_VISIBLE_DEVICES=6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMNIGIBSON_NO_SIGNALS=1 uv run scripts/train_val.py pi0_b1k_2nd \
    --exp_name="$EXP_NAME" \
    --overwrite \
    --batch_size=64 \
    --weight_loader.params_path=gs://openpi-assets/checkpoints/pi0_base/params \
    --num_train_steps=50000 \
    --val_log_interval=3000
