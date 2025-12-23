#!/bin/bash
# Monitor progress of all 3 Lambda instances in real-time

LLAMA_HOST="150.136.112.32"
GEMMA_HOST="158.101.120.239"
QWEN_HOST="150.136.209.214"
USER="ubuntu"

while true; do
    clear
    echo "=================================="
    echo "LAMBDA INSTANCES LIVE PROGRESS"
    echo "=================================="
    date
    echo ""

    echo "--- LLAMA-3.1-70B ($LLAMA_HOST) ---"
    ssh -o ConnectTimeout=5 $USER@$LLAMA_HOST "wc -l ~/empathy-probes/data/contrastive_pairs/generation_progress_llama-70b.jsonl 2>/dev/null || echo '0 pairs generated'" 2>/dev/null || echo "Unable to connect"
    ssh -o ConnectTimeout=5 $USER@$LLAMA_HOST "tail -3 ~/empathy-probes/llama70b.log 2>/dev/null" 2>/dev/null || echo ""
    echo ""

    echo "--- GEMMA-2-27B ($GEMMA_HOST) ---"
    ssh -o ConnectTimeout=5 $USER@$GEMMA_HOST "wc -l ~/empathy-probes/data/contrastive_pairs/generation_progress_gemma-27b.jsonl 2>/dev/null || echo '0 pairs generated'" 2>/dev/null || echo "Unable to connect"
    ssh -o ConnectTimeout=5 $USER@$GEMMA_HOST "tail -3 ~/empathy-probes/gemma27b.log 2>/dev/null" 2>/dev/null || echo ""
    echo ""

    echo "--- QWEN-2.5-32B ($QWEN_HOST) ---"
    ssh -o ConnectTimeout=5 $USER@$QWEN_HOST "wc -l ~/empathy-probes/data/contrastive_pairs/generation_progress_qwen-32b.jsonl 2>/dev/null || echo '0 pairs generated'" 2>/dev/null || echo "Unable to connect"
    ssh -o ConnectTimeout=5 $USER@$QWEN_HOST "tail -3 ~/empathy-probes/qwen32b.log 2>/dev/null" 2>/dev/null || echo ""
    echo ""

    echo "=================================="
    echo "Refreshing every 30 seconds..."
    echo "Press Ctrl+C to exit"
    echo "=================================="

    sleep 30
done
