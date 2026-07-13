#!/bin/bash
# rescue3c RESUME: stepA (norm_matched_controls) already completed by the
# original run_rescue3c.sh launch; this runs the remaining steps with the
# QA-corrected code (norm-matched e28b nulls + direction controls, e18c gate,
# e22 stale-axis scope, LB1/LB2 exactness fixes, LB4 reframe).
# Launched automatically by resume_watcher.sh after stepA's python exits and
# safe_pull.sh lands the corrected commit. Do NOT rerun stepA.
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
    || { echo "[rescue3c-resume] CUDA unavailable before $1 — aborting"; exit 1; }
}
step () { echo "[rescue3c-resume] $(date) $1"; }

[ -f results/norm_matched_resid_gemma/norm_matched_controls.json ] \
  || { echo "[rescue3c-resume] stepA output missing — refusing to run"; exit 1; }

cuda_ok stepC; step stepC
"$CONDA" run -n empathy python -u src/e18_interaction.py \
  --direction $DIR --writers $WRITERS --suppressors $SUPPS \
  --random-components $RAND \
  --cells cost_axis=$V22/cost_axis_templated.jsonl \
          nonsocial_axis=$V22/nonsocial_axis_templated.jsonl \
          need_mild=$V22/need_mild_axis_templated.jsonl \
          need_resolved=$V22/need_resolved_axis_templated.jsonl \
  --out results/e21_need_resid_gemma \
  > r3c_c.log 2>&1 || { echo "[rescue3c-resume] stepC FAILED"; exit 1; }
"$CONDA" run -n empathy python -u src/analysis/e18c_slope_interaction.py \
  --run results/e21_need_resid_gemma/e18.json \
  --out results/e21_need_resid_gemma \
  >> r3c_c.log 2>&1 || { echo "[rescue3c-resume] stepC analysis FAILED"; exit 1; }

cuda_ok stepB; step stepB
"$CONDA" run -n empathy python -u src/e28b_slope_nulls.py \
  --direction $DIR --suppressors $SUPPS \
  --n-sets 24 --pool-size 6 --n-random-directions 8 \
  --out results/e28b_slope_nulls_gemma \
  > r3c_b.log 2>&1 || { echo "[rescue3c-resume] stepB FAILED"; exit 1; }

cuda_ok stepLB1; step stepLB1
"$CONDA" run -n empathy python -u src/analysis/behavioral_dla.py \
  --out results/lb1_behavioral_dla_gemma \
  > r3c_lb1.log 2>&1 || { echo "[rescue3c-resume] stepLB1 FAILED"; exit 1; }

cuda_ok stepLB2; step stepLB2
"$CONDA" run -n empathy python -u src/analysis/logit_lens_trajectory.py \
  --direction $DIR --writers $WRITERS --suppressors $SUPPS \
  --out results/lb2_logit_lens_gemma \
  > r3c_lb2.log 2>&1 || { echo "[rescue3c-resume] stepLB2 FAILED"; exit 1; }

step stepLB3
"$CONDA" run -n empathy python -u src/analysis/svd_alignment.py \
  --direction $DIR --out results/lb3_svd_alignment_gemma \
  > r3c_lb3.log 2>&1 || { echo "[rescue3c-resume] stepLB3 FAILED"; exit 1; }

cuda_ok stepD; step stepD
"$CONDA" run -n empathy python -u src/e18_interaction.py \
  --direction $DIR --writers $WRITERS --suppressors $SUPPS \
  --random-components $RAND \
  --cells moral=$V22/moral_axis_templated.jsonl \
  --out results/e22_moral_resid_gemma \
  > r3c_d.log 2>&1 || { echo "[rescue3c-resume] stepD FAILED"; exit 1; }
echo "[rescue3c-resume] stepD is a STALE-AXIS descriptive replication (moral_axis_v1); it does NOT clear E22b"
"$CONDA" run -n empathy python -u src/analysis/e22_analysis.py \
  --run results/e22_moral_resid_gemma/e18.json \
  --out results/e22_moral_resid_gemma \
  >> r3c_d.log 2>&1 || { echo "[rescue3c-resume] stepD analysis FAILED"; exit 1; }

cuda_ok stepLB4; step stepLB4
if [ ! -d "$HOME/juan/jacobian-lens" ]; then
  git clone https://github.com/anthropics/jacobian-lens "$HOME/juan/jacobian-lens" \
    >> r3c_lb4.log 2>&1
fi
"$CONDA" run -n empathy pip install -e "$HOME/juan/jacobian-lens" >> r3c_lb4.log 2>&1
"$CONDA" run -n empathy python -u src/analysis/jlens_workspace.py \
  --direction $DIR --fit-prompts 300 --out results/lb4_jlens_gemma \
  >> r3c_lb4.log 2>&1 || echo "[rescue3c-resume] stepLB4 FAILED (exploratory, non-fatal)"

echo "[rescue3c-resume] $(date) RESCUE3C_DONE"
