#!/bin/bash

# EXP_NAME="openpi_05_$(date +%Y%m%d_%H%M%S)"
EXP_NAME="openpi_05_20251113_045215_resume"
echo "Experiment name: $EXP_NAME"

# Download checkpoint from S3 from original folder into new training recovery folder
CKPT_S3_PATH="s3://behavior-challenge/outputs/checkpoints/pi05_b1k_22_TASKS_oversample/openpi_05_20251113_045215/83000"
CKPT_LOCAL_PATH="outputs/checkpoints/pi05_b1k_22_TASKS_oversample/openpi_05_20251113_045215_resume/83000"
aws s3 sync $CKPT_S3_PATH $CKPT_LOCAL_PATH

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMNIGIBSON_NO_SIGNALS=1 uv run scripts/train_val.py pi05_b1k_22_TASKS_oversample \
    --exp_name="$EXP_NAME" \
    --resume \
    --batch_size=256 \
    --weight_loader.params_path=gs://openpi-assets/checkpoints/pi05_base/params \
    --force_new_wandb_run \
    --val_log_interval=3000
