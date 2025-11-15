#!/bin/bash

CHECKPOINT_FOLDER=outputs/checkpoints/pi05_b1k_22_TASKS_oversample/openpi_05_20251113_045215/81000/
S3_CHECKPOINT_FOLDER=s3://behavior-challenge/$CHECKPOINT_FOLDER
aws s3 sync "$S3_CHECKPOINT_FOLDER" "$CHECKPOINT_FOLDER"

EXP_NAME="openpi_05_$(date +%Y%m%d_%H%M%S)"
echo "Experiment name: $EXP_NAME"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMNIGIBSON_NO_SIGNALS=1 uv run scripts/train_val.py pi05_b1k_oversample_mbts \
    --exp_name="$EXP_NAME" \
    --overwrite \
    --batch_size=128 \
    --weight_loader.params_path="$CHECKPOINT_FOLDER/params" \
    --num_train_steps=100000 \
    --val_log_interval=3000
