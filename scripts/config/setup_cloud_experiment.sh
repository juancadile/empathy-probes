#!/bin/bash
# Setup script for vast.ai/Colab cloud experiments
# Tests 3 models: Qwen2.5-7B, Dolphin-Llama-3.1-8B, GPT-oss-20b

set -e  # Exit on error

echo "=== Empathy Probes Cloud Experiment Setup ==="
echo "Models: Qwen2.5-7B, Dolphin-Llama-3.1-8B, GPT-oss-20b"
echo ""

# Check if we're on Colab or vast.ai
if [ -d "/content" ]; then
    echo "Detected Google Colab environment"
    WORKDIR="/content/empathy-action-probes"
else
    echo "Detected local/vast.ai environment"
    WORKDIR="$HOME/empathy-action-probes"
fi

# Install dependencies
echo "[1/5] Installing dependencies..."
pip install -q torch transformers accelerate bitsandbytes datasets scikit-learn numpy pandas matplotlib seaborn

# Clone repo if not exists (Colab)
if [ ! -d "$WORKDIR" ]; then
    echo "[2/5] Cloning repository..."
    cd $(dirname $WORKDIR)
    git clone https://github.com/YOUR_USERNAME/empathy-action-probes.git
    cd $WORKDIR
else
    echo "[2/5] Repository already exists, skipping clone"
    cd $WORKDIR
fi

# Download models (will use HF cache)
echo "[3/5] Pre-downloading models to HuggingFace cache..."
python3 << 'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

models = [
    "Qwen/Qwen2.5-7B-Instruct",
    "cognitivecomputations/dolphin-2.9.3-llama-3.1-8b",  # Uncensored Llama
    "openai/gpt-oss-20b"
]

for model_name in models:
    print(f"\nDownloading {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Don't load full model yet, just cache the files
        print(f"✓ {model_name} cached successfully")
    except Exception as e:
        print(f"✗ Failed to cache {model_name}: {e}")
EOF

echo "[4/5] Checking GPU availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo "[5/5] Creating results directory..."
mkdir -p results/cross_model_validation

echo ""
echo "=== Setup Complete ==="
echo "Ready to run experiments!"
echo ""
echo "Next steps:"
echo "  1. Run probe extraction: python src/probe_extraction_cross_model.py"
echo "  2. Run steering tests: python src/steering_cross_model.py"
echo "  3. Analyze results: python src/analyze_cross_model.py"
