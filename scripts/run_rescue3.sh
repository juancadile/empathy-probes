#!/bin/bash
# rescue3: corrected rescue steps + new-set revalidation (Sol review 2026-07-12)
#   step5  : matched nulls, FIXED writer arm (2 MLPs, was wrongly 1 MLP + 1 head)
#   step4b : format stress rerun with T control under ALL assays
#   step5b : held-out certification of d_resid (M_confirm / T_confirm)
#   step6  : game variants, length-matched pools + resolved-explicit-first
#   step7  : capability (MMLU + wikitext) for the NEW component sets
#   step8  : E18 cost/nonsocial interaction for the NEW component sets
# Run on Spark from ~/juan/empathy-probes:
#   nohup bash scripts/run_rescue3.sh > rescue3.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
DIR=results/controlled_directions_gemma2_9b_it/direction_M_resid_block20.npy
WRITERS="L20MLP,L19MLP"
SUPPS="L18H13,L20H10,L19H12,L17H7"
RAND="L1MLP,L2MLP,L18H8,L20H5,L19H7,L17H2"

cuda_ok () {
  conda run -n empathy python -c "import torch; assert torch.cuda.is_available()" \
    || { echo "[rescue3] CUDA unavailable before $1 — aborting"; exit 1; }
}
step () { echo "[rescue3] $(date) $1"; }

cuda_ok step5; step step5
conda run -n empathy python -u src/e26_matched_nulls.py \
  --direction $DIR --writers $WRITERS --suppressors $SUPPS \
  --n-sets 12 --out results/e26_matched_nulls_gemma \
  > e26_mn3.log 2>&1 || { echo "[rescue3] step5 FAILED"; exit 1; }

cuda_ok step4b; step step4b
conda run -n empathy python -u src/e26_format_stress.py \
  --direction $DIR --writers $WRITERS --suppressors $SUPPS \
  --random-components $RAND --out results/e26_format_stress_gemma \
  > e26_fs3.log 2>&1 || { echo "[rescue3] step4b FAILED"; exit 1; }

cuda_ok step5b; step step5b
conda run -n empathy python -u src/analysis/certify_direction_heldout.py \
  --direction $DIR --block 20 \
  --out results/controlled_directions_gemma2_9b_it/e25_heldout_cert.json \
  > e25_cert.log 2>&1 || { echo "[rescue3] step5b FAILED"; exit 1; }

cuda_ok step6; step step6
conda run -n empathy python -u src/eia_validation/e27_game_variants.py \
  --direction $DIR --suppressors $SUPPS --out results/e27_game_variants \
  > e27_games.log 2>&1 || { echo "[rescue3] step6 FAILED"; exit 1; }

cuda_ok step7; step step7
conda run -n empathy python -u src/capability_eval.py \
  --direction $DIR --writers $WRITERS --suppressors $SUPPS \
  --out results/capability_eval_resid_gemma \
  > e28_capability.log 2>&1 || { echo "[rescue3] step7 FAILED"; exit 1; }

cuda_ok step8; step step8
conda run -n empathy python -u src/e18_interaction.py \
  --direction $DIR --writers $WRITERS --suppressors $SUPPS \
  --out results/e18_interaction_resid_gemma \
  > e28_e18.log 2>&1 || { echo "[rescue3] step8 FAILED"; exit 1; }

echo "[rescue3] $(date) RESCUE3_DONE"
