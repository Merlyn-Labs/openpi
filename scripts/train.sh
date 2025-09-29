    # --overwrite \
    # --exp_name="openpi_$(date +%Y%m%d_%H%M%S)" \
    
    # 2>&1 | tee "openpi_20250919_180225_resume/$(date +%Y%m%d_%H%M%S).log"

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_val.py pi05_b1k \
#     --exp_name="openpi_05_$(date +%Y%m%d_%H%M%S)" \
#     --batch_size=16 \
#     --num_train_steps=100000

EXP_NAME="openpi_$(date +%Y%m%d_%H%M%S)"
echo "Experiment name: $EXP_NAME"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_val.py pi0_b1k \
    --exp_name="$EXP_NAME" \
    --overwrite \
    --batch_size=16 \
    --num_train_steps=100000
