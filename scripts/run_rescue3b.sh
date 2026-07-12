#!/bin/bash
# rescue3b: resume after codex review of rescue3 (steps 1-5b done and kept).
#   step5c : matched nulls WITH overlap sets -> full C(8,2)=28 writer rank test
#   step6  : game variants, FIXED seed lifecycle (fresh LocalPlayer per run)
#   step7  : capability (MMLU + wikitext) for the NEW component sets
#   step8  : E18 interaction for the NEW sets + composition-matched random
# Run on Spark from ~/juan/empathy-probes:
#   nohup bash scripts/run_rescue3b.sh > rescue3b.log 2>&1 < /dev/null &
set -u
CONDA="${CONDA:-$HOME/miniforge3/bin/conda}"
cd "$(dirname "$0")/.."
DIR=results/controlled_directions_gemma2_9b_it/direction_M_resid_block20.npy
WRITERS="L20MLP,L19MLP"
SUPPS="L18H13,L20H10,L19H12,L17H7"
RAND="L1MLP,L2MLP,L18H8,L20H5,L19H7,L17H2"

cuda_ok () {
  "$CONDA" run -n empathy python -c "import torch; assert torch.cuda.is_available()" \
    || { echo "[rescue3b] CUDA unavailable before $1 — aborting"; exit 1; }
}
step () { echo "[rescue3b] $(date) $1"; }

cuda_ok step5c; step step5c
"$CONDA" run -n empathy python -u src/e26_matched_nulls.py \
  --direction $DIR --writers $WRITERS --suppressors $SUPPS \
  --n-sets 12 --include-overlap --out results/e26_matched_nulls_gemma \
  > e26_mn3b.log 2>&1 || { echo "[rescue3b] step5c FAILED"; exit 1; }

cuda_ok step6; step step6
rm -rf results/e27_game_variants  # partial histories from the killed seed-lifecycle-buggy run
"$CONDA" run -n empathy python -u src/eia_validation/e27_game_variants.py \
  --direction $DIR --suppressors $SUPPS --out results/e27_game_variants \
  > e27_games.log 2>&1 || { echo "[rescue3b] step6 FAILED"; exit 1; }

cuda_ok step7; step step7
"$CONDA" run -n empathy python -u src/capability_eval.py \
  --direction $DIR --writers $WRITERS --suppressors $SUPPS \
  --out results/capability_eval_resid_gemma \
  > e28_capability.log 2>&1 || { echo "[rescue3b] step7 FAILED"; exit 1; }

cuda_ok step8; step step8
"$CONDA" run -n empathy python -u src/e18_interaction.py \
  --direction $DIR --writers $WRITERS --suppressors $SUPPS \
  --random-components $RAND \
  --out results/e18_interaction_resid_gemma \
  > e28_e18.log 2>&1 || { echo "[rescue3b] step8 FAILED"; exit 1; }

echo "[rescue3b] $(date) RESCUE3B_DONE"
