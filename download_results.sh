#!/bin/bash
# Download results from all 3 Lambda instances

LLAMA_HOST="150.136.112.32"
GEMMA_HOST="158.101.120.239"
QWEN_HOST="150.136.209.214"
USER="ubuntu"
LOCAL_DIR="data/contrastive_pairs"

echo "=================================="
echo "DOWNLOADING RESULTS FROM LAMBDA"
echo "=================================="
echo ""

# Create local directory if needed
mkdir -p "$LOCAL_DIR"

# Download Llama-70B results
echo "Downloading Llama-3.1-70B results..."
scp $USER@$LLAMA_HOST:~/empathy-probes/data/contrastive_pairs/generation_progress_llama-70b.jsonl "$LOCAL_DIR/" 2>/dev/null && echo "✓ Llama-70B downloaded" || echo "✗ Llama-70B not found"

# Download Gemma-27B results
echo "Downloading Gemma-2-27B results..."
scp $USER@$GEMMA_HOST:~/empathy-probes/data/contrastive_pairs/generation_progress_gemma-27b.jsonl "$LOCAL_DIR/" 2>/dev/null && echo "✓ Gemma-27B downloaded" || echo "✗ Gemma-27B not found"

# Download Qwen-32B results
echo "Downloading Qwen-2.5-32B results..."
scp $USER@$QWEN_HOST:~/empathy-probes/data/contrastive_pairs/generation_progress_qwen-32b.jsonl "$LOCAL_DIR/" 2>/dev/null && echo "✓ Qwen-32B downloaded" || echo "✗ Qwen-32B not found"

echo ""
echo "=================================="
echo "DOWNLOAD COMPLETE"
echo "=================================="
echo ""

# Show counts
echo "Pair counts:"
for file in "$LOCAL_DIR"/generation_progress_{llama-70b,gemma-27b,qwen-32b}.jsonl; do
    if [ -f "$file" ]; then
        count=$(wc -l < "$file" | tr -d ' ')
        echo "  $(basename $file): $count pairs"
    fi
done

echo ""
echo "Files saved to: $LOCAL_DIR/"
