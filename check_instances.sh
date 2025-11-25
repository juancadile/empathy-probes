#!/bin/bash
# Quick check if all Lambda instances are ready

LLAMA_HOST="150.136.112.32"
GEMMA_HOST="158.101.120.239"
QWEN_HOST="150.136.209.214"
USER="ubuntu"

echo "=================================="
echo "CHECKING LAMBDA INSTANCES"
echo "=================================="
echo ""

check_instance() {
    local name=$1
    local host=$2

    echo -n "Checking $name ($host)... "

    if ssh -o ConnectTimeout=5 -o BatchMode=yes $USER@$host "echo 'OK'" 2>/dev/null | grep -q "OK"; then
        echo "✅ READY"
        return 0
    else
        echo "❌ NOT READY (still booting or network issue)"
        return 1
    fi
}

check_instance "Llama-70B" "$LLAMA_HOST"
check_instance "Gemma-27B" "$GEMMA_HOST"
check_instance "Qwen-32B" "$QWEN_HOST"

echo ""
echo "=================================="
echo "If all instances are READY, you can proceed with:"
echo "  ./launch_llama70b.sh"
echo "  ./launch_gemma27b.sh"
echo "  ./launch_qwen32b.sh"
echo ""
echo "If NOT READY, wait 1-2 minutes and run this script again."
echo "=================================="
