#!/bin/bash

export CUDA_VISIBLE_DEVICES=6;

export TRAIN_CONFIG_NAME="pi05_b1k_oversample_hee";
export CKPT_NAME="hee_openpi_05_20251116_064228";
export STEP_COUNT=18000;
export TASK_NAME="hiding_Easter_eggs";

# export CONTROL_MODE="receeding_horizon";
# export MAX_LEN=100;
# export ACTION_HORIZON=100;
# export TEMPORAL_ENSEMBLE_MAX=1;
# export EXP_K_VALUE=1.0;

export CONTROL_MODE="receeding_temporal";
export MAX_LEN=72;
export ACTION_HORIZON=12;
export TEMPORAL_ENSEMBLE_MAX=6;
export EXP_K_VALUE=0.5;

aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/${TRAIN_CONFIG_NAME}/${CKPT_NAME}/${STEP_COUNT}/ \
    /workspace/openpi/outputs/checkpoints/${TRAIN_CONFIG_NAME}/${CKPT_NAME}/${STEP_COUNT}/

# conda activate behavior
kill $(lsof -ti:8002)
XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/serve_b1k.py \
    --port 8002 \
    --task_name=$TASK_NAME \
    --control_mode=$CONTROL_MODE \
    --max_len=$MAX_LEN \
    --action_horizon=$ACTION_HORIZON \
    --temporal_ensemble_max=$TEMPORAL_ENSEMBLE_MAX \
    --exp_k_value=$EXP_K_VALUE \
    policy:checkpoint \
    --policy.config=pi05_b1k_inference_final \
    --policy.dir=/workspace/openpi/outputs/checkpoints/$TRAIN_CONFIG_NAME/$CKPT_NAME/$STEP_COUNT/
