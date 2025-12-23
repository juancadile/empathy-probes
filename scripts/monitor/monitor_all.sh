#!/bin/bash
# Monitor all 3 Lambda instances in one view

LLAMA_HOST="150.136.112.32"
GEMMA_HOST="158.101.120.239"
QWEN_HOST="150.136.209.214"
USER="ubuntu"

echo "=================================="
echo "ALL MODELS LIVE MONITOR"
echo "=================================="
echo "Press Ctrl+C to exit (doesn't stop generation)"
echo ""

while true; do
    clear
    echo "================================================================================"
    echo "                      LAMBDA GENERATION LIVE MONITOR"
    echo "================================================================================"
    date
    echo ""

    # Llama-70B
    echo "┌────────────────────────────────────────────────────────────────────────────┐"
    echo "│ LLAMA-3.1-70B ($LLAMA_HOST)                                    │"
    echo "└────────────────────────────────────────────────────────────────────────────┘"

    LLAMA_PAIRS=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $USER@$LLAMA_HOST "wc -l ~/empathy-probes/data/contrastive_pairs/generation_progress_llama-70b.jsonl 2>/dev/null | awk '{print \$1}'" 2>/dev/null || echo "0")
    LLAMA_PERCENT=$((LLAMA_PAIRS * 100 / 500))
    LLAMA_FILLED=$((LLAMA_PAIRS / 10))
    LLAMA_EMPTY=$((50 - LLAMA_FILLED))

    echo "Progress: $LLAMA_PAIRS / 500 pairs ($LLAMA_PERCENT%)"
    printf "["
    printf "%${LLAMA_FILLED}s" | tr ' ' '█'
    printf "%${LLAMA_EMPTY}s" | tr ' ' '░'
    printf "]\n"

    echo "Recent activity:"
    ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $USER@$LLAMA_HOST "tail -3 ~/empathy-probes/llama70b.log 2>/dev/null" 2>/dev/null | sed 's/^/  /' || echo "  No log available"
    echo ""

    # Gemma-27B
    echo "┌────────────────────────────────────────────────────────────────────────────┐"
    echo "│ GEMMA-2-27B ($GEMMA_HOST)                                      │"
    echo "└────────────────────────────────────────────────────────────────────────────┘"

    GEMMA_PAIRS=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $USER@$GEMMA_HOST "wc -l ~/empathy-probes/data/contrastive_pairs/generation_progress_gemma-27b.jsonl 2>/dev/null | awk '{print \$1}'" 2>/dev/null || echo "0")
    GEMMA_PERCENT=$((GEMMA_PAIRS * 100 / 500))
    GEMMA_FILLED=$((GEMMA_PAIRS / 10))
    GEMMA_EMPTY=$((50 - GEMMA_FILLED))

    echo "Progress: $GEMMA_PAIRS / 500 pairs ($GEMMA_PERCENT%)"
    printf "["
    printf "%${GEMMA_FILLED}s" | tr ' ' '█'
    printf "%${GEMMA_EMPTY}s" | tr ' ' '░'
    printf "]\n"

    echo "Recent activity:"
    ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $USER@$GEMMA_HOST "tail -3 ~/empathy-probes/gemma27b.log 2>/dev/null" 2>/dev/null | sed 's/^/  /' || echo "  No log available"
    echo ""

    # Qwen-32B
    echo "┌────────────────────────────────────────────────────────────────────────────┐"
    echo "│ QWEN-2.5-32B ($QWEN_HOST)                                      │"
    echo "└────────────────────────────────────────────────────────────────────────────┘"

    QWEN_PAIRS=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $USER@$QWEN_HOST "wc -l ~/empathy-probes/data/contrastive_pairs/generation_progress_qwen-32b.jsonl 2>/dev/null | awk '{print \$1}'" 2>/dev/null || echo "0")
    QWEN_PERCENT=$((QWEN_PAIRS * 100 / 500))
    QWEN_FILLED=$((QWEN_PAIRS / 10))
    QWEN_EMPTY=$((50 - QWEN_FILLED))

    echo "Progress: $QWEN_PAIRS / 500 pairs ($QWEN_PERCENT%)"
    printf "["
    printf "%${QWEN_FILLED}s" | tr ' ' '█'
    printf "%${QWEN_EMPTY}s" | tr ' ' '░'
    printf "]\n"

    echo "Recent activity:"
    ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $USER@$QWEN_HOST "tail -3 ~/empathy-probes/qwen32b.log 2>/dev/null" 2>/dev/null | sed 's/^/  /' || echo "  No log available"
    echo ""

    # Overall summary
    TOTAL_PAIRS=$((LLAMA_PAIRS + GEMMA_PAIRS + QWEN_PAIRS))
    OVERALL_PERCENT=$((TOTAL_PAIRS * 100 / 1500))

    echo "================================================================================"
    echo "OVERALL: $TOTAL_PAIRS / 1500 pairs ($OVERALL_PERCENT% complete)"
    echo "================================================================================"
    echo "Refreshing every 15 seconds..."

    sleep 15
done
