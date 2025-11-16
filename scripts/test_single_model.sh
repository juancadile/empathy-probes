#!/bin/bash
# Quick test: Run probe extraction on ONE model to verify setup
# Useful for testing before running expensive cloud experiments

set -e

echo "=== Testing Probe Extraction (Single Model) ==="
echo ""
echo "This will test ONE model to verify everything works."
echo "Estimated time: ~30-40 minutes"
echo ""

# Ask which model to test
echo "Which model to test?"
echo "1) qwen2.5-7b (7B params, ~14GB RAM)"
echo "2) dolphin-llama-3.1-8b (8B params, ~16GB RAM)"
echo "3) gpt-oss-20b (20B params, ~40GB RAM - needs good GPU)"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
  1)
    MODEL="qwen2.5-7b"
    ;;
  2)
    MODEL="dolphin-llama-3.1-8b"
    ;;
  3)
    MODEL="gpt-oss-20b"
    ;;
  *)
    echo "Invalid choice. Exiting."
    exit 1
    ;;
esac

echo ""
echo "Testing with: $MODEL"
echo ""

# Check Python dependencies
echo "[1/3] Checking dependencies..."
python3 -c "import torch; import transformers; import sklearn" 2>/dev/null || {
  echo "Missing dependencies. Installing..."
  pip install -q torch transformers accelerate scikit-learn numpy
}

# Check GPU
echo "[2/3] Checking GPU..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Run extraction
echo "[3/3] Running probe extraction..."
python3 src/probe_extraction_cross_model.py --models $MODEL

echo ""
echo "=== Test Complete ==="
echo ""
echo "Results saved to: results/cross_model_validation/${MODEL}_results.json"
echo ""
echo "Next steps:"
echo "  1. Check results: cat results/cross_model_validation/${MODEL}_results.json"
echo "  2. If successful, run all models: python src/probe_extraction_cross_model.py --models all"
echo "  3. Or upload to Colab/vast.ai for faster execution"
