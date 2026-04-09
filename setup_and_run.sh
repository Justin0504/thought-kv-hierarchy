#!/bin/bash
# === Week 1 Validation Setup & Run Script ===
# Usage: bash setup_and_run.sh
# Run this on the GPU server after scp-ing the project folder

set -e

echo "=== Step 1: Create conda environment ==="
conda create -n thought-hbm python=3.10 -y 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate thought-hbm

pip install torch transformers accelerate datasets numpy pandas matplotlib seaborn scipy tqdm

echo ""
echo "=== Step 2: Check GPU ==="
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
echo ""

echo "=== Step 3: Quick model download test ==="
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', trust_remote_code=True)
print(f'Tokenizer loaded OK, vocab size: {tok.vocab_size}')
"

echo ""
echo "=== Step 4: Run validation (50 samples on cuda:0) ==="
python scripts/run_week1_validation.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --n_samples 50 \
    --device cuda:0 \
    --output_dir results/week1

echo ""
echo "=== Done! Check results/week1/ ==="
ls -la results/week1/
