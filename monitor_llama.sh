#!/bin/bash
# Monitor Llama-3.1-70B generation progress

HOST="150.136.112.32"
USER="ubuntu"

echo "=================================="
echo "LLAMA-3.1-70B LIVE MONITOR"
echo "=================================="
echo "Instance: $HOST"
echo "Press Ctrl+C to exit (doesn't stop generation)"
echo ""

while true; do
    clear
    echo "=================================="
    echo "LLAMA-3.1-70B ($HOST)"
    echo "=================================="
    date
    echo ""

    # Check if process is running
    echo "--- Process Status ---"
    ssh -o ConnectTimeout=5 $USER@$HOST "ps aux | grep -v grep | grep generate_opensource_vllm.py" 2>/dev/null || echo "Process not running"
    echo ""

    # Count generated pairs
    echo "--- Progress ---"
    PAIRS=$(ssh -o ConnectTimeout=5 $USER@$HOST "wc -l ~/empathy-probes/data/contrastive_pairs/generation_progress_llama-70b.jsonl 2>/dev/null | awk '{print \$1}'" 2>/dev/null)
    if [ -z "$PAIRS" ]; then
        PAIRS=0
    fi
    PERCENT=$((PAIRS * 100 / 500))
    echo "Generated: $PAIRS / 500 pairs ($PERCENT%)"

    # Progress bar
    FILLED=$((PAIRS / 10))
    EMPTY=$((50 - FILLED))
    printf "["
    printf "%${FILLED}s" | tr ' ' '='
    printf "%${EMPTY}s" | tr ' ' '-'
    printf "]\n"
    echo ""

    # Show last 10 lines of log
    echo "--- Recent Log (last 10 lines) ---"
    ssh -o ConnectTimeout=5 $USER@$HOST "tail -10 ~/empathy-probes/llama70b.log 2>/dev/null" 2>/dev/null || echo "Log not available"
    echo ""

    echo "=================================="
    echo "Refreshing every 10 seconds..."
    echo "=================================="

    sleep 10
done
