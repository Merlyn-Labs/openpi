#!/bin/bash

    # --overwrite \
    # --exp_name="openpi_$(date +%Y%m%d_%H%M%S)" \
    
    # 2>&1 | tee "openpi_20250919_180225_resume/$(date +%Y%m%d_%H%M%S).log"

EXP_NAME="openpi_05_$(date +%Y%m%d_%H%M%S)"
echo "Experiment name: $EXP_NAME"
HF_HOME=/scratch/huggingface/ XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 uv run scripts/train_val.py pi05_b1k \
    --exp_name="$EXP_NAME" \
    --overwrite \
    --batch_size=256 \
    --weight_loader.params_path=/root/openpi/outputs/checkpoints/pi05_b1k/openpi_05_20251023_221110/28000/params \
    --num_train_steps=50000
