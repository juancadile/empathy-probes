#!/bin/bash
# State-vs-trait battery over the V2 directions of record (DFA-derived).
# NOTE: results/probes/empathy_direction_layer_*.npy are the V1 Phi-3-mini
# probes (d_model 3072) — do NOT use them with gemma/llama.
# Run on the Spark (CUDA). See notes/EXPERIMENT_LOG.md 2026-07-20 entry.
#
# Usage: nohup bash scripts/run_state_trait_battery.sh > state_trait_battery.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

CONDA="${CONDA:-$HOME/miniforge3/bin/conda}"
"$CONDA" run -n empathy python -c "import torch; assert torch.cuda.is_available()" \
    || { echo "CUDA unavailable in empathy env" >&2; exit 1; }

run () {  # model direction layer
    echo "=== $1 :: $2 (L$3) ==="
    "$CONDA" run -n empathy python -u src/state_trait_battery.py \
        --model "$1" --direction "$2" --layer "$3" \
        --transcripts data/state_trait_transcripts.jsonl \
        --n-null 50 --seed 0
}

# gemma2-9b-it (d_model 3584)
run gemma2_9b_it results/dfa_gemma2_9b_it/empathy_direction_layer8.npy 8
run gemma2_9b_it results/dfa_gemma2_9b_it_L20/empathy_direction_layer20.npy 20
run gemma2_9b_it results/dfa_gemma2_9b_it_M_block20/empathy_direction_layer20.npy 20

# llama31-8b-it (d_model 4096)
run llama31_8b_it results/dfa_llama31_8b_it/empathy_direction_layer15.npy 15

echo "battery complete: results/state_trait_battery_*/"
