#!/bin/bash

# EXP_NAME="openpi_05_20251112_214833"

CHECKPOINT_FOLDER=outputs/checkpoints/pi05_b1k_22_TASKS_oversample/openpi_05_20251113_045215/48000/
S3_CHECKPOINT_FOLDER=s3://behavior-challenge/$CHECKPOINT_FOLDER
aws s3 sync "$S3_CHECKPOINT_FOLDER" "$CHECKPOINT_FOLDER"

# echo "Waiting for 45k checkpoint to appear on S3..."
# while true; do
#     # Check for the existence of params file in S3 checkpoint directory
#     if aws s3 ls "$S3_CHECKPOINT_FOLDER/params" > /dev/null 2>&1; then
#         echo "Found 45k checkpoint on S3, syncing now..."
#         # sleep 15m  # Wait for 15 minutes to make sure the checkpoint is fully synced to S3
#         aws s3 sync "$S3_CHECKPOINT_FOLDER" "$CHECKPOINT_FOLDER"
#         if [ -f "$CHECKPOINT_FOLDER/params" ]; then
#             echo "Checkpoint successfully synced locally."
#             break
#         else
#             echo "Checkpoint not yet synced locally. Retrying sync..."
#         fi
#     else
#         echo "Checkpoint not found yet, waiting 5 minutes..."
#         sleep 5m
#     fi
# done

# EXP_NAME="openpi_05_$(date +%Y%m%d_%H%M%S)"
EXP_NAME="openpi_05_20251114_055221"
echo "Experiment name: $EXP_NAME"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMNIGIBSON_NO_SIGNALS=1 uv run scripts/train_val.py pi05_b1k_oversample \
    --exp_name="$EXP_NAME" \
    --resume \
    --batch_size=128 \
    --weight_loader.params_path="$CHECKPOINT_FOLDER/params" \
    --num_train_steps=75000 \
    --val_log_interval=3000
