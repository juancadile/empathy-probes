#!/bin/bash
# rescue3c: remaining blockers (codex-final-review priority a > c > b > d)
#           + lens & attribution battery LB1-LB4 (roadmap, user request)
#   stepA   : fresh norm-matched random-DIRECTION controls, new sets,
#             CONFIRM M/T, no stale reference, 20 draws
#   stepC   : fresh E21 need axis (all arms, one invocation) + e18c analysis
#             with the pre-registered paired urgent-resolved slope contrast
#   stepB   : E18 slope null over 24 layer-multiset-matched four-head sets
#             + 8 same-head random-direction slope controls
#   stepLB1 : true direct logit attribution (behavioral)
#   stepLB2 : per-layer logit-lens trajectory (baseline + edited)
#   stepLB3 : SVD alignment of direction vs component weights (CPU)
#   stepD   : E22 moral-vs-moral for new sets + TOST analysis
#   stepLB4 : Jacobian-lens workspace membership (exploratory; non-fatal)
# Run on Spark from ~/juan/empathy-probes:
#   nohup bash scripts/run_rescue3c.sh > rescue3c.log 2>&1 < /dev/null &
set -u
CONDA="${CONDA:-$HOME/miniforge3/bin/conda}"
cd "$(dirname "$0")/.."
DIR=results/controlled_directions_gemma2_9b_it/direction_M_resid_block20.npy
WRITERS="L20MLP,L19MLP"
SUPPS="L18H13,L20H10,L19H12,L17H7"
RAND="L1MLP,L2MLP,L18H8,L20H5,L19H7,L17H2"
V22=data/contrastive_pairs/v2_2

cuda_ok () {
  "$CONDA" run -n empathy python -c "import torch; assert torch.cuda.is_available()" \
    || { echo "[rescue3c] CUDA unavailable before $1 — aborting"; exit 1; }
}
step () { echo "[rescue3c] $(date) $1"; }

cuda_ok stepA; step stepA
"$CONDA" run -n empathy python -u src/norm_matched_controls.py \
  --direction $DIR --writers $WRITERS --suppressors $SUPPS \
  --m-pairs data/contrastive_pairs/v2_1/M_confirm_templated.jsonl \
  --t-pairs data/contrastive_pairs/v2_1/T_confirm_templated.jsonl \
  --reference "" --n-seeds 20 --out results/norm_matched_resid_gemma \
  > r3c_a.log 2>&1 || { echo "[rescue3c] stepA FAILED"; exit 1; }

cuda_ok stepC; step stepC
"$CONDA" run -n empathy python -u src/e18_interaction.py \
  --direction $DIR --writers $WRITERS --suppressors $SUPPS \
  --random-components $RAND \
  --cells cost_axis=$V22/cost_axis_templated.jsonl \
          nonsocial_axis=$V22/nonsocial_axis_templated.jsonl \
          need_mild=$V22/need_mild_axis_templated.jsonl \
          need_resolved=$V22/need_resolved_axis_templated.jsonl \
  --out results/e21_need_resid_gemma \
  > r3c_c.log 2>&1 || { echo "[rescue3c] stepC FAILED"; exit 1; }
"$CONDA" run -n empathy python -u src/analysis/e18c_slope_interaction.py \
  --run results/e21_need_resid_gemma/e18.json \
  --out results/e21_need_resid_gemma \
  >> r3c_c.log 2>&1 || { echo "[rescue3c] stepC analysis FAILED"; exit 1; }

cuda_ok stepB; step stepB
"$CONDA" run -n empathy python -u src/e28b_slope_nulls.py \
  --direction $DIR --suppressors $SUPPS \
  --n-sets 24 --n-random-directions 8 --out results/e28b_slope_nulls_gemma \
  > r3c_b.log 2>&1 || { echo "[rescue3c] stepB FAILED"; exit 1; }

cuda_ok stepLB1; step stepLB1
"$CONDA" run -n empathy python -u src/analysis/behavioral_dla.py \
  --out results/lb1_behavioral_dla_gemma \
  > r3c_lb1.log 2>&1 || { echo "[rescue3c] stepLB1 FAILED"; exit 1; }

cuda_ok stepLB2; step stepLB2
"$CONDA" run -n empathy python -u src/analysis/logit_lens_trajectory.py \
  --direction $DIR --writers $WRITERS --suppressors $SUPPS \
  --out results/lb2_logit_lens_gemma \
  > r3c_lb2.log 2>&1 || { echo "[rescue3c] stepLB2 FAILED"; exit 1; }

step stepLB3
"$CONDA" run -n empathy python -u src/analysis/svd_alignment.py \
  --direction $DIR --out results/lb3_svd_alignment_gemma \
  > r3c_lb3.log 2>&1 || { echo "[rescue3c] stepLB3 FAILED"; exit 1; }

cuda_ok stepD; step stepD
"$CONDA" run -n empathy python -u src/e18_interaction.py \
  --direction $DIR --writers $WRITERS --suppressors $SUPPS \
  --random-components $RAND \
  --cells moral=$V22/moral_axis_templated.jsonl \
  --out results/e22_moral_resid_gemma \
  > r3c_d.log 2>&1 || { echo "[rescue3c] stepD FAILED"; exit 1; }
"$CONDA" run -n empathy python -u src/analysis/e22_analysis.py \
  --run results/e22_moral_resid_gemma/e18.json \
  --out results/e22_moral_resid_gemma \
  >> r3c_d.log 2>&1 || { echo "[rescue3c] stepD analysis FAILED"; exit 1; }

cuda_ok stepLB4; step stepLB4
if [ ! -d "$HOME/juan/jacobian-lens" ]; then
  git clone https://github.com/anthropics/jacobian-lens "$HOME/juan/jacobian-lens" \
    >> r3c_lb4.log 2>&1
fi
"$CONDA" run -n empathy pip install -e "$HOME/juan/jacobian-lens" >> r3c_lb4.log 2>&1
"$CONDA" run -n empathy python -u src/analysis/jlens_workspace.py \
  --direction $DIR --fit-prompts 300 --out results/lb4_jlens_gemma \
  >> r3c_lb4.log 2>&1 || echo "[rescue3c] stepLB4 FAILED (exploratory, non-fatal)"

echo "[rescue3c] $(date) RESCUE3C_DONE"
