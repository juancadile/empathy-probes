#!/bin/bash
# Finetuning trait-axis (beta) test on the Spark. Stages are resumable —
# rerunning skips completed LoRA checkpoints; earlier stage outputs are
# plain JSON that later stages read.
#
# Usage: nohup bash scripts/run_finetune_trait_test.sh > finetune_trait_test.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

CONDA="${CONDA:-$HOME/miniforge3/bin/conda}"
RUN () { "$CONDA" run -n empathy python -u src/finetune_trait_test.py "$@"; }

"$CONDA" run -n empathy python -c "import torch, peft; assert torch.cuda.is_available()" \
    || { echo "CUDA or peft unavailable in empathy env" >&2; exit 1; }

echo "=== build-data ==="; RUN build-data
echo "=== gen-base ===";   RUN gen-base
echo "=== delta-p ===";    RUN delta-p
echo "=== finetune ===";   RUN finetune
echo "=== posttest ===";   RUN posttest
echo "=== analyze ===";    RUN analyze
echo "finetune trait test complete: results/finetune_trait_test_gemma2_9b_it/"
