# Experiment Log — V2 Mechanistic Investigation

**Canonical chronological record.** One entry per experiment: date, script/commit, data files, headline numbers, interpretation, caveats. Newest at the bottom. Keep this updated with every run — interpretations here are the claim-language of record (deliberately narrowed wordings are authoritative).

Model under study unless noted: **google/gemma-2-9b-it**, on the DGX Spark. Assistant attribution: [C] = Claude, [S] = Sol (OpenAI), [C/S] = joint.

---

## 2026-07-10

### E01 · Direct Feature Attribution, shallow readout (L*=8) [C]
- **Script:** `src/direct_feature_attribution.py` · **Data:** `results/dfa_gemma2_9b_it/dfa_summary.json`, `empathy_direction_layer8.npy` · **Issue:** #25
- 300 pairs from the confounded V2 dataset; layer sweep picked block 8/42, test AUROC 0.997. Exactness checks passed (head-sum 5e-3, residual reconstruction 7e-4).
- Semi-sparse: 16/153 components carry 50% of paired attribution, 47 carry 80%. Top writers L8MLP, L5MLP, block-8 heads. Embedding stream = **10.7%** of separation.
- **Interpretation (revised by E04/E05):** this direction is substantially a lexical echo; treat as historical.

### E02 · DFA, mid-network readout (L*=20) [C]
- **Data:** `results/dfa_gemma2_9b_it_L20/` · **Issue:** #25
- AUROC 0.978; cos(d_L8, d_L20) = **0.45** — the directions are different objects. Embed share 3.1%. Writers are late heads (L17–L20) in a **push–pull** pattern (5 of top-12 negative). Zero overlap with L8 top-20.

### E03 · V2.1 stimulus suite generated [C]
- **Scripts:** `src/data_generation/generate_v2_1_suite.py`, `generate_v2_1_claude_batch.py`, `build_cell_m.py` · **Data:** `data/contrastive_pairs/v2_1/`
- Cells A,B,D,E,F,G,H,S × {claude-haiku (Batches API), gpt-4o-mini, gemini-2.5-flash}; cell M (40 templated matched-lexicon pairs); cell R (60 perturbations). ~$2.50 total. Design: `notes/v2_1-stimulus-suite-design.md` (prediction matrix). [S] later added cell T (matched task persistence) and API-varied M/T expansions + label audit (E10).

### E04 · Lexical stress battery, tests 1–2 [C]
- **Script:** `src/lexical_battery.py` · **Data:** `results/lexical_battery/lexical_battery.json` · **Issue:** #33
- **Shuffled-token control: AUROC 0.989 (L8) / 0.901 (L20)** — the V2 contrastive dataset is lexically saturated; headline AUROCs on it are not evidence of behavioral representation. Embedding-direction regression barely dents AUROC (lexical content recombines nonlinearly). Logit lens: negative pole of both directions decodes to task/efficiency vocabulary ("operations, objectives, efficiency…"; efficiency in 5 languages at L20); positive pole includes "feeling/feels" → direction reads as **feelings-vs-efficiency** at the vocabulary level.

### E05 · Cell M probe test (test 3) [C]
- **Script:** `src/cell_m_probe_test.py` · **Data:** `results/lexical_battery/cell_m.json` · **Issue:** #33
- On matched-lexicon pairs: L8 direction ≈ chance (full 0.564; decision-tokens 0.658). **L20 direction: decision-tokens AUROC 0.799**, paired diff concentrated at the decision clause (6.59 vs 1.95 full-text).
- **Interpretation:** two-level story — V1's early probe read vocabulary; a genuine decision representation exists deeper, weaker than headline numbers suggested.

### E06 · TransformerLens ↔ HF bridge check [C]
- **Log:** `~/juan/bridge_check.log` (Spark, not committed) · `empathy` conda env (aarch64)
- cos(TL, HF) = 1.000 at blocks 8 and 20 → TL validated for interventions on this stack.

---

## 2026-07-11 (Sol session — see ROADMAP Stage A for full wording)

### E07 · Component mean-ablation with cell-M metrics (A2) [C/S]
- **Script:** `src/activation_patching.py` (batched by [S]) · **Data:** `results/patching_gemma2_9b_it*/` (incl. `_M_block20`, `_cell_t` variants)
- Ablations of L18H3, L20H7 cut controlled separation (>7σ beyond random); L8MLP, L19H5 cut separation *and* helping preference. **DFA writer strength only weakly predicts causal necessity (r=0.26)** with the old probe. Some components (incl. a random L19MLP draw) move behavior without moving the probe → no component is empathy-specific under the old direction.

### E08 · Controlled (purified) direction + transfer profile [S]
- **Data:** `results/controlled_directions_gemma2_9b_it/direction_M_block20.npy`, layerwise transfer analyses
- Cell-M-trained block-20 direction transfers to costly helping (**A 0.726**) and third-person (**F 0.908**; reciprocal F→M 0.975) but is quiet on task-only (E 0.519), motive (G 0.546), warmth (D 0.244), caring character (H 0.165), matched task persistence (**T 0.047**). After ~block 24 it absorbs warmth/persona content.
- **Interpretation:** the mechanistic target is **costly helping / wellbeing-over-task action at block 20**; the original A-direction is a broad warmth/persona axis. Substantially resolves Fork 2.

### E09 · Purified DFA (block 20, cell-M direction) [S]
- **Data:** `results/dfa_gemma2_9b_it_M_block20/dfa_summary.json`
- Purified writer strength predicts representation necessity **r=0.57** (vs 0.26 for old probe). 56 components carry 80% → moderately distributed mechanism.

### E10 · V2.1 explicit-label audit [S]
- **Data:** `results/v2_1_label_audit/`
- 342/528 pairs accepted (370 distinguishable; 92.4% intended polarity among those). Cell B's contrast largely collapses when helping is free.

### E11 · Component taxonomy / specificity gate (A2.5) [S]
- Policy-tradeoff: L19MLP, L17H13, L18H13, L20H15 · helping-selective: L19H12, L15H15 · generic decision machinery: L15/L16/L11 MLPs, L19H5 · representation-only writers: L20MLP, L18H3, L20H7.

### E12 · Inside-component inspection + path mediation [S]
- **Data:** `results/action_component_inspection_gemma2_9b_it/`, `results/action_path_mediation*/`
- Action heads in top ~5% of static OV write gain for the purified direction (L18H13 99.1st pct). L18H3 = behaviorally inert context reader. L19MLP weight alignment diffuse (top-50 channels 2.2%) → SAE features preferred over neuron claims.
- Restoration-confirmed partial paths: **L17H13→L17MLP (75% rescue), L15H15→L19MLP (42%), L19MLP→L20MLP (38%)** (bootstrap CIs exclude 0). L18H13/L19H12 causal but **not mediated by tested targets** — competing explanations: other consumers, direct residual effects, redundancy, off-manifold restoration. (Claim deliberately narrowed by Juan — do not re-inflate.)

### E13 · Targeted weight orthogonalization, bidirectional [S] — commit 2ed963d
- **Script:** `src/weight_orthogonalization.py` · **Data:** `results/weight_orthogonalization_gemma2_9b_it/analysis.json`
- **Positive writers (L19MLP+L20H15) removed: helping −0.206 [−0.259, −0.156]**, task −0.022 (CI incl. 0). **Suppressor-direction weights removed: helping +0.263 [0.167, 0.377]**. Random 6-component edits: −0.073. Neutral KL < 0.001, 100% top-token agreement.
- **Interpretation:** parameter-level causal evidence for a **push–pull costly-helping policy**. Publication gates at that point: norm-matched controls, real capability benchmark, EIA-harness validation, cross-model replication.

### E14 · Norm-matched random controls (gate 1) [C]
- **Script:** `src/norm_matched_controls.py` · **Data:** `results/norm_matched_controls_gemma2_9b_it/norm_matched_controls.json`
- Same components, delta-norm-matched, random directions, 10 seeds/set: null −0.0045 ± 0.0044 (positive writers) and −0.0126 ± 0.0037 (suppressors) → targeted effects at **z = −45 / +75**. Equal-magnitude random damage to the same weights does ~nothing.
- **Caveat:** run's baseline helping (1.09) differs from pilot print (4.28) — loading/aggregation difference unresolved; script now recomputes targeted deltas in-run (rerun queued) so z-scores are like-with-like. Conclusion robust to any plausible rescaling.

### E15 · Capability benchmark under edits (gate 2) [C] — RERUNNING (MMLU fix)
- **Script:** `src/capability_eval.py` · **Data:** `results/capability_eval_gemma2_9b_it/` (`capability_eval_broken_mmlu.json` = first attempt)
- 400 MMLU + wikitext-2 ppl under baseline / positive-writers / suppressors / targeted-k6.
- **Perplexity result (valid from first run): edits move wikitext ppl ≤ 0.4%** (13.096 baseline → 13.082 / 13.135 / 13.131). Consistent with selectivity.
- **MMLU harness bug caught:** raw-completion letter readout was degenerate — 0.2575 (exactly chance) identically across all conditions, i.e. a constant letter bias making predictions edit-invariant. Diagnostic tell: *identical* accuracy across conditions is a red flag, not a selectivity result. Fixed with chat-template prompting + logsumexp over space-variant letter tokens; rerunning. Prior crash: `datasets` 5.x requires `Salesforce/wikitext`.

### E16 · EIA-harness behavioral validation (gate 3) [C] — IN PREPARATION
- Original EIA game harness vendored to `third_party/eia/empathy/` (MIT). Plan: monkeypatch its single LLM entrypoint (`call_llm_with_prompt`) → local Gemma under {baseline, positive-writers, suppressors} edits; 5 scenarios × 3 seeds; judge calls (hard-coded `provider="openai"`) deferred to offline scoring (Batches API) + rule-based scores from `experiment.json` histories.
- **The decisive question:** does the weight-edited model actually help less (or more, for suppressor edits) in the original *games* — actions, not text style.

### Remaining gates
- E17 (planned): cross-model replication on Llama-3.1-8B.
