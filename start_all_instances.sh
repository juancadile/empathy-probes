#!/bin/bash
# Automated launcher for all 3 instances

set -e

LLAMA_HOST="150.136.112.32"
GEMMA_HOST="158.101.120.239"
QWEN_HOST="150.136.209.214"
USER="ubuntu"

echo "=================================="
echo "STARTING ALL INSTANCES"
echo "=================================="
echo ""

# Function to setup and start generation on an instance
start_instance() {
    local name=$1
    local host=$2
    local model=$3
    local logfile=$4

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Setting up: $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Upload files
    echo "1. Uploading files..."
    scp -o ConnectTimeout=10 data/eia_scenarios/scenarios.json $USER@$host:~/empathy-probes/data/eia_scenarios/ 2>/dev/null || { echo "❌ Upload failed"; return 1; }
    scp -o ConnectTimeout=10 src/generate_opensource_vllm.py $USER@$host:~/empathy-probes/src/ 2>/dev/null || { echo "❌ Upload failed"; return 1; }
    scp -o ConnectTimeout=10 lambda_setup.sh $USER@$host:~/empathy-probes/ 2>/dev/null || { echo "❌ Upload failed"; return 1; }
    echo "   ✓ Files uploaded"

    # Run setup script
    echo "2. Running setup (installing vLLM, may take 2-3 minutes)..."
    ssh -o ConnectTimeout=10 $USER@$host "cd ~/empathy-probes && bash lambda_setup.sh" || { echo "❌ Setup failed"; return 1; }
    echo "   ✓ Setup complete"

    # Start generation in background
    echo "3. Starting generation..."
    ssh -o ConnectTimeout=10 $USER@$host "cd ~/empathy-probes/src && nohup python3 generate_opensource_vllm.py --model $model > ../$logfile 2>&1 &" || { echo "❌ Start failed"; return 1; }
    echo "   ✓ Generation started"

    echo "✅ $name is now running!"
    echo ""
}

# Start all instances
start_instance "Llama-3.1-70B" "$LLAMA_HOST" "llama-3.1-70b-instruct" "llama70b.log" &
start_instance "Gemma-2-27B" "$GEMMA_HOST" "gemma-2-27b-it" "gemma27b.log" &
start_instance "Qwen-2.5-32B" "$QWEN_HOST" "qwen-2.5-32b-instruct" "qwen32b.log" &

# Wait for all to complete
wait

echo "=================================="
echo "ALL INSTANCES STARTED"
echo "=================================="
echo ""
echo "Monitor progress with:"
echo "  ./monitor_all.sh       (all 3 in one view)"
echo "  ./monitor_llama.sh     (Llama only)"
echo "  ./monitor_gemma.sh     (Gemma only)"
echo "  ./monitor_qwen.sh      (Qwen only)"
echo ""
echo "Download results with:"
echo "  ./download_results.sh"
echo ""
