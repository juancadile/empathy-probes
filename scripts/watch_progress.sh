#!/bin/bash
# Live progress monitor for dataset generation with detailed breakdown

while true; do
    clear
    echo "================================================================================"
    echo "                    DATASET GENERATION LIVE PROGRESS"
    echo "================================================================================"
    echo ""

    # Target counts
    TOTAL_TARGET=2500
    TARGET_PER_MODEL=500  # 5 scenarios × 100 pairs

    # Count per-model progress files
    echo "Per-Model Progress:"
    echo "--------------------------------------------------------------------------------"

    TOTAL_NEW=0

    for model_file in data/contrastive_pairs/generation_progress_*.jsonl; do
        if [ -f "$model_file" ]; then
            model_name=$(basename "$model_file" .jsonl | sed 's/generation_progress_//')
            count=$(wc -l < "$model_file" 2>/dev/null | tr -d ' ')
            percent=$(echo "scale=1; $count * 100 / $TARGET_PER_MODEL" | bc 2>/dev/null || echo "0")
            TOTAL_NEW=$((TOTAL_NEW + count))

            # Progress bar
            filled=$(echo "$count * 40 / $TARGET_PER_MODEL" | bc 2>/dev/null || echo "0")
            printf "%-15s [" "$model_name"
            for i in $(seq 1 40); do
                if [ $i -le $filled ]; then
                    printf "="
                else
                    printf " "
                fi
            done
            printf "] %4d/%d (%5.1f%%)\n" "$count" "$TARGET_PER_MODEL" "$percent"
        fi
    done

    echo ""
    echo "Overall Progress:"
    echo "--------------------------------------------------------------------------------"

    # Calculate total unique pairs
    TOTAL_PAIRS=$TOTAL_NEW
    TOTAL_PERCENT=$(echo "scale=1; $TOTAL_PAIRS * 100 / $TOTAL_TARGET" | bc 2>/dev/null || echo "0")
    REMAINING=$((TOTAL_TARGET - TOTAL_PAIRS))

    echo "Total pairs generated: $TOTAL_PAIRS / $TOTAL_TARGET"
    echo "Completion: $TOTAL_PERCENT%"
    echo "Remaining: $REMAINING pairs"

    # Overall progress bar
    total_filled=$(echo "$TOTAL_PAIRS * 60 / $TOTAL_TARGET" | bc 2>/dev/null || echo "0")
    printf "\nOverall: ["
    for i in $(seq 1 60); do
        if [ $i -le $total_filled ]; then
            printf "="
        else
            printf " "
        fi
    done
    printf "] %5.1f%%\n" "$TOTAL_PERCENT"

    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "Last updated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Press Ctrl+C to exit monitoring"
    echo "================================================================================"

    sleep 3
done
