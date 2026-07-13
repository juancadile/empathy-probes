#!/bin/bash
# rescue3c RESUME 2: post R1-correction chain. StepA's original artifact
# (results/norm_matched_resid_gemma) is PROVISIONAL — independent per-component
# random vectors + theoretical-only norms — and is PRESERVED, not overwritten;
# the corrected R1 rerun writes to results/norm_matched_resid_gemma_v2.
# Never reruns valid stepC raw output (e18.json) — but a skip requires
# validation (scripts/validate_stepc_e18.py: conditions_spec == current sets,
# all four cells present, provenance hashes == current direction/data files);
# an INVALID artifact ABORTS the chain rather than silently skipping or
# overwriting. e18c analysis reruns if missing OR produced by the pre-
# mean-effect version of the script. LB3 (SVD, exploratory) runs AFTER stepD
# and is non-fatal so its failure/time risk cannot block D. Does NOT deploy or
# kill any process; at the end it commits and pushes ONLY the result/log paths
# it produced (unrelated/untracked files are left alone).
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
    || { echo "[rescue3c-resume2] CUDA unavailable before $1 — aborting"; exit 1; }
}
step () { echo "[rescue3c-resume2] $(date) $1"; }

# --- stepC: NEVER rerun valid raw output; a skip REQUIRES validation ---
if [ -f results/e21_need_resid_gemma/e18.json ]; then
  step "stepC raw output present — validating before skip"
  "$CONDA" run -n empathy python -u scripts/validate_stepc_e18.py \
    --run results/e21_need_resid_gemma/e18.json \
    --direction $DIR --writers $WRITERS --suppressors $SUPPS --random $RAND \
    --cells cost_axis=$V22/cost_axis_templated.jsonl \
            nonsocial_axis=$V22/nonsocial_axis_templated.jsonl \
            need_mild=$V22/need_mild_axis_templated.jsonl \
            need_resolved=$V22/need_resolved_axis_templated.jsonl \
    || { echo "[rescue3c-resume2] stepC e18.json INVALID for the current chain config — ABORTING (will not silently skip or overwrite)"; exit 1; }
  step "stepC raw output validated — NOT rerunning"
else
  cuda_ok stepC; step "stepC raw"
  "$CONDA" run -n empathy python -u src/e18_interaction.py \
    --direction $DIR --writers $WRITERS --suppressors $SUPPS \
    --random-components $RAND \
    --cells cost_axis=$V22/cost_axis_templated.jsonl \
            nonsocial_axis=$V22/nonsocial_axis_templated.jsonl \
            need_mild=$V22/need_mild_axis_templated.jsonl \
            need_resolved=$V22/need_resolved_axis_templated.jsonl \
    --out results/e21_need_resid_gemma \
    > r3c_c.log 2>&1 || { echo "[rescue3c-resume2] stepC FAILED"; exit 1; }
fi
if [ -f results/e21_need_resid_gemma/e18c.json ] && \
   "$CONDA" run -n empathy python -c "import json,sys; d=json.load(open('results/e21_need_resid_gemma/e18c.json')); sys.exit(0 if all('mean_effect_by_need' in c and 'mean_effect_urgent_minus_resolved_paired' in c for c in d['conditions'].values()) else 1)"; then
  step "stepC e18c analysis present (with need-level mean effects) — skipping"
else
  step "stepC e18c analysis (missing or pre-mean-effect version — rerunning)"
  "$CONDA" run -n empathy python -u src/analysis/e18c_slope_interaction.py \
    --run results/e21_need_resid_gemma/e18.json \
    --out results/e21_need_resid_gemma \
    >> r3c_c.log 2>&1 || { echo "[rescue3c-resume2] stepC analysis FAILED"; exit 1; }
fi

# --- corrected R1: shared random vector per seed + realized-norm 3% gate ---
cuda_ok stepR1; step "stepR1 (corrected norm-matched controls)"
"$CONDA" run -n empathy python -u src/norm_matched_controls.py \
  --direction $DIR --writers $WRITERS --suppressors $SUPPS \
  --m-pairs data/contrastive_pairs/v2_1/M_confirm_templated.jsonl \
  --t-pairs data/contrastive_pairs/v2_1/T_confirm_templated.jsonl \
  --reference "" --n-seeds 20 --out results/norm_matched_resid_gemma_v2 \
  > r3c_r1.log 2>&1 || { echo "[rescue3c-resume2] stepR1 FAILED"; exit 1; }

# --- corrected stepB: realized norms persisted + gated; theoretical keys renamed ---
cuda_ok stepB; step stepB
"$CONDA" run -n empathy python -u src/e28b_slope_nulls.py \
  --direction $DIR --suppressors $SUPPS \
  --n-sets 24 --pool-size 6 --n-random-directions 8 \
  --out results/e28b_slope_nulls_gemma \
  > r3c_b.log 2>&1 || { echo "[rescue3c-resume2] stepB FAILED"; exit 1; }

cuda_ok stepLB1; step stepLB1
"$CONDA" run -n empathy python -u src/analysis/behavioral_dla.py \
  --out results/lb1_behavioral_dla_gemma \
  > r3c_lb1.log 2>&1 || { echo "[rescue3c-resume2] stepLB1 FAILED"; exit 1; }

cuda_ok stepLB2; step stepLB2
"$CONDA" run -n empathy python -u src/analysis/logit_lens_trajectory.py \
  --direction $DIR --writers $WRITERS --suppressors $SUPPS \
  --out results/lb2_logit_lens_gemma \
  > r3c_lb2.log 2>&1 || { echo "[rescue3c-resume2] stepLB2 FAILED"; exit 1; }

cuda_ok stepD; step stepD
"$CONDA" run -n empathy python -u src/e18_interaction.py \
  --direction $DIR --writers $WRITERS --suppressors $SUPPS \
  --random-components $RAND \
  --cells moral=$V22/moral_axis_templated.jsonl \
  --out results/e22_moral_resid_gemma \
  > r3c_d.log 2>&1 || { echo "[rescue3c-resume2] stepD FAILED"; exit 1; }
echo "[rescue3c-resume2] stepD is a STALE-AXIS descriptive replication (moral_axis_v1); it does NOT clear E22b"
"$CONDA" run -n empathy python -u src/analysis/e22_analysis.py \
  --run results/e22_moral_resid_gemma/e18.json \
  --out results/e22_moral_resid_gemma \
  >> r3c_d.log 2>&1 || { echo "[rescue3c-resume2] stepD analysis FAILED"; exit 1; }

# LB3 (randomized truncated SVD, --device auto with CPU fallback) is
# EXPLORATORY: it runs AFTER stepD and its failure/time risk is NON-FATAL
step "stepLB3 (exploratory, non-fatal)"
"$CONDA" run -n empathy python -u src/analysis/svd_alignment.py \
  --direction $DIR --device auto --out results/lb3_svd_alignment_gemma \
  > r3c_lb3.log 2>&1 || echo "[rescue3c-resume2] stepLB3 FAILED (exploratory, non-fatal)"

cuda_ok stepLB4; step "stepLB4 (exploratory, non-fatal)"
if [ ! -d "$HOME/juan/jacobian-lens" ]; then
  git clone https://github.com/anthropics/jacobian-lens "$HOME/juan/jacobian-lens" \
    >> r3c_lb4.log 2>&1
fi
"$CONDA" run -n empathy pip install -e "$HOME/juan/jacobian-lens" >> r3c_lb4.log 2>&1
"$CONDA" run -n empathy python -u src/analysis/jlens_workspace.py \
  --direction $DIR --fit-prompts 300 --out results/lb4_jlens_gemma \
  >> r3c_lb4.log 2>&1 || echo "[rescue3c-resume2] stepLB4 FAILED (exploratory, non-fatal)"

# --- commit + push results only; never `git add -A` (preserve unrelated/untracked) ---
step "commit+push results"
git add \
  results/e21_need_resid_gemma \
  results/norm_matched_resid_gemma_v2 \
  results/e28b_slope_nulls_gemma \
  results/lb1_behavioral_dla_gemma \
  results/lb2_logit_lens_gemma \
  results/lb3_svd_alignment_gemma \
  results/e22_moral_resid_gemma \
  results/lb4_jlens_gemma 2>/dev/null
git commit -m "rescue3c resume2: corrected R1 (shared rand vec + realized-norm gate), corrected stepB, LB1-LB4, stale-axis D" \
  || echo "[rescue3c-resume2] nothing to commit"
git push || echo "[rescue3c-resume2] push failed — push manually"

echo "[rescue3c-resume2] $(date) RESCUE3C_RESUME2_DONE"
