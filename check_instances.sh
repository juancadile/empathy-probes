#!/bin/bash
# Comprehensive check of Lambda instances - models, processes, and output files

LLAMA_HOST="150.136.112.32"
GEMMA_HOST="158.101.120.239"
QWEN_HOST="150.136.209.214"
USER="ubuntu"

echo "================================================================================"
echo "                    LAMBDA INSTANCE HEALTH CHECK"
echo "================================================================================"
echo ""

check_instance() {
    local name=$1
    local host=$2
    local model_name=$3
    local output_file=$4

    echo "┌────────────────────────────────────────────────────────────────────────────┐"
    echo "│ $name ($host)"
    echo "└────────────────────────────────────────────────────────────────────────────┘"

    # Check SSH connectivity
    echo -n "  [1/6] SSH connectivity... "
    if ssh -o ConnectTimeout=5 -o BatchMode=yes $USER@$host "echo 'OK'" 2>/dev/null | grep -q "OK"; then
        echo "✓ Connected"
    else
        echo "✗ Failed - instance not reachable"
        echo ""
        return 1
    fi

    # Check if model is downloaded
    echo -n "  [2/6] Model download ($model_name)... "
    model_check=$(ssh $USER@$host "
        if python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('$model_name', local_files_only=True)\" &>/dev/null; then
            model_path=\$(python3 -c \"from huggingface_hub import snapshot_download; print(snapshot_download('$model_name', local_files_only=True))\" 2>/dev/null)
            size=\$(du -sh \"\$model_path\" 2>/dev/null | cut -f1)
            echo \"EXISTS:\$size\"
        else
            echo \"MISSING\"
        fi
    " 2>/dev/null)

    if [[ $model_check == EXISTS:* ]]; then
        size=$(echo $model_check | cut -d: -f2)
        echo "✓ Downloaded ($size)"
    else
        echo "✗ Not found - model may still be downloading"
    fi

    # Check if generation process is running
    echo -n "  [3/6] Generation process... "
    process_check=$(ssh $USER@$host "pgrep -f 'generate_contrastive_pairs_lambda.py' > /dev/null && echo 'RUNNING' || echo 'STOPPED'" 2>/dev/null)
    if [ "$process_check" == "RUNNING" ]; then
        echo "✓ Running"
    else
        echo "✗ Not running"
    fi

    # Check output file status
    echo -n "  [4/6] Output file ($output_file)... "
    file_check=$(ssh $USER@$host "
        if [ -f empathy-action-probes/$output_file ]; then
            lines=\$(wc -l < empathy-action-probes/$output_file)
            age_sec=\$(( \$(date +%s) - \$(stat -c %Y empathy-action-probes/$output_file) ))
            echo \"EXISTS:\$lines:\$age_sec\"
        else
            echo \"MISSING\"
        fi
    " 2>/dev/null)

    if [[ $file_check == EXISTS:* ]]; then
        IFS=':' read -ra INFO <<< "$file_check"
        lines=${INFO[1]}
        age_sec=${INFO[2]}
        age_min=$((age_sec / 60))

        if [ $age_sec -lt 600 ]; then
            echo "✓ Active ($lines pairs, updated ${age_min}m ago)"
        else
            echo "⚠ Stale ($lines pairs, updated ${age_min}m ago)"
        fi
    else
        echo "✗ Not found"
    fi

    # Check disk space
    echo -n "  [5/6] Disk space... "
    disk_usage=$(ssh $USER@$host "df -h / | tail -1 | awk '{print \$5}'" 2>/dev/null)
    if [ -n "$disk_usage" ]; then
        echo "✓ $disk_usage used"
    else
        echo "✗ Could not check"
    fi

    # Check GPU
    echo -n "  [6/6] GPU status... "
    gpu_check=$(ssh $USER@$host "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1" 2>/dev/null)
    if [ -n "$gpu_check" ]; then
        IFS=',' read -ra GPU <<< "$gpu_check"
        util=$(echo ${GPU[0]} | xargs)
        mem_used=$(echo ${GPU[1]} | xargs)
        mem_total=$(echo ${GPU[2]} | xargs)
        echo "✓ ${util}% util, ${mem_used}/${mem_total}MB"
    else
        echo "⚠ Could not check"
    fi

    echo ""
}

check_instance "Llama-3.1-70B" "$LLAMA_HOST" "meta-llama/Llama-3.1-70B-Instruct" "data/contrastive_pairs/generation_progress_llama-3.1-70b.jsonl"
check_instance "Gemma-2-27B" "$GEMMA_HOST" "google/gemma-2-27b-it" "data/contrastive_pairs/generation_progress_gemma-2-27b.jsonl"
check_instance "Qwen-2.5-32B" "$QWEN_HOST" "Qwen/Qwen2.5-32B-Instruct" "data/contrastive_pairs/generation_progress_qwen-2.5-32b.jsonl"

echo "================================================================================"
echo "Health check complete!"
echo "================================================================================"
