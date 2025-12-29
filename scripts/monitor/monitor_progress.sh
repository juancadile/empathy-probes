#!/bin/bash
# Simple progress monitor for dataset generation

PROGRESS_FILE="data/contrastive_pairs/generation_progress.jsonl"
TOTAL=2500

while true; do
    clear
    echo "=================================="
    echo "DATASET GENERATION PROGRESS"
    echo "=================================="
    echo ""

    if [ -f "$PROGRESS_FILE" ]; then
        COUNT=$(wc -l < "$PROGRESS_FILE" | tr -d ' ')
        PERCENT=$(echo "scale=2; $COUNT * 100 / $TOTAL" | bc)

        echo "Completed: $COUNT / $TOTAL pairs ($PERCENT%)"
        echo ""
        echo "Progress bar:"
        FILLED=$(echo "$COUNT * 50 / $TOTAL" | bc)
        printf "["
        for i in $(seq 1 50); do
            if [ $i -le $FILLED ]; then
                printf "="
            else
                printf " "
            fi
        done
        printf "] $PERCENT%%\n"
        echo ""
        echo "Last updated: $(date)"
    else
        echo "Progress file not found yet..."
    fi

    echo ""
    echo "Press Ctrl+C to exit monitoring"
    sleep 5
done
