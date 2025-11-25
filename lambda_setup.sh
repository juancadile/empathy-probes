#!/bin/bash
# Lambda Labs Instance Setup Script
# Clones repo on v2-expansion branch, installs deps, downloads model

set -e

MODEL=${1:-""}  # Pass model as first argument
UPLOADED_FILES=${2:-"false"}  # Whether files were already uploaded

echo "=================================="
echo "LAMBDA INSTANCE SETUP"
echo "=================================="
echo "Model: $MODEL"
echo "=================================="
echo ""

# Check GPU
echo "[1/6] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Update system
echo "[2/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y git python3-pip python3-venv -qq
echo "✓ System packages installed"
echo ""

# Create project structure
echo "[3/6] Setting up project structure..."
if [ "$UPLOADED_FILES" == "true" ]; then
    if [ -d "empathy-action-probes" ]; then
        cd empathy-action-probes
        echo "  ✓ Using uploaded files"
    else
        echo "  ✗ Error: empathy-action-probes directory not found"
        exit 1
    fi
else
    echo "  ✗ Error: Files must be uploaded first"
    exit 1
fi
echo ""

# Install Python dependencies
echo "[4/6] Installing Python dependencies (vLLM, torch, etc.)..."
echo "  This may take 2-3 minutes..."
pip3 install --upgrade pip -q
pip3 install vllm torch transformers accelerate huggingface_hub -q
echo "✓ Python dependencies installed"
echo ""

# Pre-download model if specified
if [ -n "$MODEL" ]; then
    echo "[5/6] Pre-downloading model: $MODEL"
    echo "  This may take 10-30 minutes depending on model size..."
    echo "  You can monitor disk usage with: watch -n 5 df -h"
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', cache_dir='/home/ubuntu/.cache/huggingface')"
    echo "✓ Model downloaded"
else
    echo "[5/6] Skipping model pre-download (no model specified)"
fi
echo ""

echo "[6/6] Creating output directories..."
mkdir -p data/contrastive_pairs
echo "✓ Setup complete"
echo ""

echo "=================================="
echo "READY TO GENERATE"
echo "=================================="
echo ""
echo "Next steps:"
echo "  cd empathy-action-probes/src"
echo "  nohup python3 generate_opensource_vllm.py --model <model-name> > ../generation.log 2>&1 &"
echo ""
echo "Monitor with:"
echo "  tail -f empathy-action-probes/generation.log"
echo ""
