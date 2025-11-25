#!/bin/bash
# Quick status check - one-time snapshot

LLAMA_HOST="150.136.112.32"
GEMMA_HOST="158.101.120.239"
QWEN_HOST="150.136.209.214"
USER="ubuntu"

echo "=================================="
echo "QUICK STATUS CHECK"
echo "=================================="
date
echo ""

check_status() {
    local name=$1
    local host=$2
    local file=$3

    echo "[$name]"
    PAIRS=$(ssh -o ConnectTimeout=5 $USER@$host "wc -l ~/empathy-probes/data/contrastive_pairs/$file 2>/dev/null | awk '{print \$1}'" 2>/dev/null || echo "0")
    RUNNING=$(ssh -o ConnectTimeout=5 $USER@$host "ps aux | grep -v grep | grep generate_opensource_vllm.py" 2>/dev/null)

    if [ ! -z "$RUNNING" ]; then
        echo "  Status: 🟢 Running"
    else
        echo "  Status: 🔴 Not running"
    fi

    PERCENT=$((PAIRS * 100 / 500))
    echo "  Progress: $PAIRS / 500 pairs ($PERCENT%)"
    echo ""
}

check_status "Llama-70B" "$LLAMA_HOST" "generation_progress_llama-70b.jsonl"
check_status "Gemma-27B" "$GEMMA_HOST" "generation_progress_gemma-27b.jsonl"
check_status "Qwen-32B" "$QWEN_HOST" "generation_progress_qwen-32b.jsonl"

TOTAL=$(ssh -o ConnectTimeout=5 $USER@$LLAMA_HOST "wc -l ~/empathy-probes/data/contrastive_pairs/generation_progress_llama-70b.jsonl 2>/dev/null | awk '{print \$1}'" 2>/dev/null || echo "0")
TOTAL=$((TOTAL + $(ssh -o ConnectTimeout=5 $USER@$GEMMA_HOST "wc -l ~/empathy-probes/data/contrastive_pairs/generation_progress_gemma-27b.jsonl 2>/dev/null | awk '{print \$1}'" 2>/dev/null || echo "0")))
TOTAL=$((TOTAL + $(ssh -o ConnectTimeout=5 $USER@$QWEN_HOST "wc -l ~/empathy-probes/data/contrastive_pairs/generation_progress_qwen-32b.jsonl 2>/dev/null | awk '{print \$1}'" 2>/dev/null || echo "0")))

echo "=================================="
echo "TOTAL: $TOTAL / 1500 pairs"
echo "=================================="
