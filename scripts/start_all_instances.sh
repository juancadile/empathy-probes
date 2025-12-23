#!/bin/bash
# Automated launcher for all 3 Lambda instances
# Sets up repo, downloads models, starts generation

set -e

LLAMA_HOST="150.136.112.32"
GEMMA_HOST="158.101.120.239"
QWEN_HOST="150.136.209.214"
USER="ubuntu"
SSH_KEY="ssh/juan.pem"

echo "=================================="
echo "STARTING ALL INSTANCES"
echo "=================================="
echo ""

# Function to setup and start generation on an instance
start_instance() {
    local name=$1
    local host=$2
    local hf_model=$3
    local short_model=$4

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Setting up: $name ($host)"
    echo "Model: $hf_model"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Upload project files
    echo "1. Uploading project files..."
    ssh -i $SSH_KEY -o StrictHostKeyChecking=no $USER@$host "mkdir -p empathy-action-probes/src empathy-action-probes/data/eia_scenarios empathy-action-probes/data/contrastive_pairs"
    scp -i $SSH_KEY -o StrictHostKeyChecking=no src/generate_opensource_vllm.py $USER@$host:empathy-action-probes/src/ || { echo "❌ Upload failed"; return 1; }
    scp -i $SSH_KEY -o StrictHostKeyChecking=no data/eia_scenarios/scenarios.json $USER@$host:empathy-action-probes/data/eia_scenarios/ || { echo "❌ Upload failed"; return 1; }
    scp -i $SSH_KEY -o StrictHostKeyChecking=no lambda_setup.sh $USER@$host:~/ || { echo "❌ Upload failed"; return 1; }
    echo "   ✓ Files uploaded"

    # Run setup script (installs deps, downloads model)
    echo "2. Running setup (this will take 15-40 minutes for model download)..."
    echo "   You can monitor progress by SSHing to the instance"
    ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10 $USER@$host "bash lambda_setup.sh $hf_model true" || { echo "❌ Setup failed"; return 1; }
    echo "   ✓ Setup complete, model downloaded"

    # Start generation in background
    echo "3. Starting generation..."
    ssh -i $SSH_KEY -o StrictHostKeyChecking=no $USER@$host "cd empathy-action-probes/src && nohup python3 generate_opensource_vllm.py --model $short_model > ../generation.log 2>&1 &" || { echo "❌ Start failed"; return 1; }
    echo "   ✓ Generation started"

    echo "✅ $name is now running!"
    echo ""
}

# Start all instances in parallel
start_instance "Llama-3.1-70B" "$LLAMA_HOST" "meta-llama/Llama-3.1-70B-Instruct" "llama-3.1-70b-instruct" &
start_instance "Gemma-2-27B" "$GEMMA_HOST" "google/gemma-2-27b-it" "gemma-2-27b-it" &
start_instance "Qwen-2.5-32B" "$QWEN_HOST" "Qwen/Qwen2.5-32B-Instruct" "qwen-2.5-32b-instruct" &

# Wait for all to complete
wait

echo "=================================="
echo "ALL INSTANCES STARTED"
echo "=================================="
echo ""
echo "Monitor progress with:"
echo "  ./check_instances.sh   (health check)"
echo "  ./monitor_all.sh       (live progress)"
echo ""
echo "Download results with:"
echo "  ./download_results.sh"
echo ""
