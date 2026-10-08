# 002-theory-update — How IOI-circuit structure scales (outline)

**Depends on 001.** Reuses `followup/001-living-update/results/` and `followup_summary.json` only —
**no new GPU jobs unless a gap must be filled.** Run via
`python -m veritas.cli.main followup <run_dir> --instruction-file <this file> --name theory-update`
→ writes `followup/002-theory-update/`.

## Question
Given 001's per-model measurements, **how does IOI-circuit structure scale?** Specifically: how do
(a) circuit size (#heads), (b) faithfulness F(C)/F(M), and (c) the presence/count of each head class
(name-mover / S-inhibition / induction / duplicate-token / previous-token / negative-name-mover) vary with
model size and generation — and can we fit a simple relation and restate the paper's claim in words?

## Prior expectation (to confirm or overturn, not assume)
The literature suggests the name-mover + S-inhibition + induction backbone is reused broadly (circuit
component reuse, arXiv 2310.08744) and that cross-scale IOI is under-studied (arXiv 2411.16105 covers only
prompt-variant generalization within GPT-2 small). So the working hypothesis is: **the backbone persists
and faithfulness stays roughly flat across scale, with head-count growing sub-linearly in params.** 001's
data decide whether that holds; report it either way, including the case where a class drops out.

## Inputs (read from disk; do not recompute originals)
- `followup/001-living-update/followup_summary.json` — per-model baseline, faithfulness, discovered
  head-lists-by-class, name-survivor counts, GPU-seconds.
- `followup/001-living-update/results/*.png` — scale figures (reuse / extend).
- Anchor originals: `run/replication/step8_completeness_result.json` (F=87.8%),
  `run/replication/step2_baseline_result.json` (logit-diff 3.49),
  `run/replication/codebase/easy_transformer/ioi_circuit_extraction.py` (26 heads / 7 classes).

## Analyses
1. **Class-persistence matrix** — rows = 6 head classes, cols = models (GPT-2 s/m/l/xl, Pythia
   70m/410m/1.4b/2.8b). Cell = #heads of that class in 001's discovered circuit (0 = class absent).
   Read off which classes are universal vs which appear/disappear with scale or generation.
2. **Scale fits** — fit and report (with R²):
   - faithfulness F(C)/F(M) vs log(params) — is it flat (survives), rising, or falling?
   - #circuit-heads vs log(params) — does the circuit grow with model size?
   - #name-mover heads and #S-inhibition heads vs log(params) separately (these are the two classes 001
     computes for every rung).
   Keep fits simple (linear in log-params, or monotone trend statement); report the coefficient and R²,
   and say plainly if the data are too few/noisy for a fit (n≈8 within-GPT-2 + Pythia).
3. **Generation contrast** — GPT-2 family (2019 arch, learned pos-emb) vs Pythia family (2023 arch,
   rotary): does faithfulness/structure differ at matched scale (e.g. gpt2-large 774M vs pythia-1.4b)?
4. **Behavioral-only rung** — fold Tier-2 (Qwen2.5-0.5B, Llama-3.2-3B, gemma-2-2b) into the *behavioral*
   survival trend only (logit-diff vs scale); explicitly NOT into the circuit-structure fits.

## Deliverables (`followup/002-theory-update/`)
- `report.md` — the class-persistence matrix; the fitted relations (coeff + R²); a generation contrast
  paragraph; and an **updated claim in words**, e.g. *"The IOI name-mover + S-inhibition backbone
  persists across GPT-2 scale and into Pythia, with faithfulness ~<X>% ± <…>; circuit head-count scales
  ~<…> with log-params; class <…> is GPT-2-specific / universal."* Every number cites 001's artifact.
- `followup_summary.json` and `result_card.json` (schema `sai.followup.result_card/v1`): headline = the
  fitted scaling statement with its key number; metrics = fit coefficients + R² + persistence counts,
  each with baseline (GPT-2-small anchor) + provenance (001 artifact); ≤2 tables (persistence matrix; fit
  table); 0–3 figures under `results/` (faithfulness-vs-scale, headcount-vs-scale, class-persistence
  heatmap).

## Anti-goals
- No new GPU jobs unless 001 left a specific, small, named gap (say which, run only that, synchronously).
- Never background and wait. Every number from 001's artifacts or this session's fit; originals read from
  disk. Disclose small-n / low-R² honestly rather than over-fitting a clean-looking curve to ~8 points.
- Do not conflate behavioral survival (Tier-2) with circuit survival (Tier-1).