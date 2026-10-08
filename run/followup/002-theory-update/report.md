# 002-theory-update — How IOI-circuit structure scales

**Depends on 001-living-update.** No new GPU jobs were run: every number below is
computed from `followup/001-living-update/`'s on-disk artifacts, with the paper's
anchors read from `run/replication/`. This study is pure analysis (numpy/scipy fits +
matplotlib) over 001's per-model measurements.

## Question

Given 001's per-model IOI measurements up the GPT-2 ladder (small→XL), across a second
generation (Pythia 70M→2.8B), and a behavioral-only modern rung (Tier-2), **how does
IOI-circuit structure scale?** Specifically, how do (a) circuit size (#heads),
(b) faithfulness F(C)/F(M), and (c) the presence/count of each of six head classes
(name-mover / S-inhibition / induction / duplicate-token / previous-token /
negative-name-mover) vary with model size and generation — can a simple relation be
fit, and how should the paper's claim be restated? The prior expectation to
confirm-or-overturn: *the name-mover + S-inhibition backbone persists broadly and
faithfulness stays roughly flat, with head-count growing sub-linearly in params.*

## Approach

- **Reused (read-only):** `followup/001-living-update/followup_summary.json` (per-model
  baseline, faithfulness, discovered head-lists-by-class, name-survivor counts) and
  `results/per_model/*.json` (per-config path-patch `pct_change` per head). Anchors read
  from `run/replication/step8_completeness_result.json` (F=87.8%),
  `step2_baseline_result.json` (logit-diff 3.49), and the paper's 26-head/7-class curated
  circuit.
- **Added:** `workspace/analyze_scaling.py` builds two class-persistence matrices, runs
  OLS fits of each quantity on log₁₀(params) with R², and computes the generation
  contrast; `workspace/make_figures.py` renders three canvas figures. Output:
  `results/fit_results.json` + three PNGs.
- **Validation:** the strong-head criterion `|pct_change/100 · default_logit_diff| ≥ 0.2`
  reproduces 001's result-card head-counts **exactly** (GPT-2-small 9/2/3/2/1/1;
  medium 11/2/1/0/0/0; large 8/2/0/0/0/0; pythia-70m 2/1/0/0/0/0; pythia-410m 1/1/0/0/0/0;
  pythia-1.4b 1/0/0/1/0/0), so the matrix I extend to all 8 models is on 001's own scale.

### One structural fact that governs the whole analysis

001's "discovered circuit" is assembled from a **fixed per-class top-k template**:
whenever discovery configs a–d all ran, the circuit is *always* 11 name-mover + 4
S-inhibition + 4 induction + 3 duplicate + 2 previous + 2 negative = **26 heads / 6
classes**, identical for every model (`discovered_circuit_counts` in each per-model JSON).
Consequently **circuit head-count and per-class membership counts do not vary across
models by construction** — the only two rungs with fewer heads (gpt2-xl = 17, pythia-2.8b
= 13) are smaller *only because 001 ran fewer configs there for budget reasons*
(gpt2-xl: configs a,b; pythia-2.8b: config a), **not** because classes were biologically
absent. I therefore report the literal template matrix for completeness but base the
scientific reading on the **strong-head** matrix (single heads that actually move the
logit-diff by ≥0.2), which does vary and is the genuine cross-scale signal.

## Results

### 1. Class-persistence matrix

**(a) Literal 001 discovered circuit (fixed template — coverage, not biology):**

| class | s | m | l | xl | 70m | 410m | 1.4b | 2.8b |
|---|---|---|---|---|---|---|---|---|
| name mover | 11 | 11 | 11 | 11 | 11 | 11 | 11 | 11 |
| s2 inhibition | 4 | 4 | 4 | 4 | 4 | 4 | 4 | n/r |
| induction | 4 | 4 | 4 | n/r | 4 | 4 | 4 | n/r |
| duplicate token | 3 | 3 | 3 | n/r | 3 | 3 | 3 | n/r |
| previous token | 2 | 2 | 2 | n/r | 2 | 2 | 2 | n/r |
| negative | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| **total** | **26** | **26** | **26** | **17** | **26** | **26** | **26** | **13** |

`n/r` = the discovery config for that class was not run in 001 (budget), distinct from a
true 0. Read literally, all six classes are "present" at every fully-discovered rung
because the template forces them to be — so this matrix answers coverage, not persistence.

**(b) Strong heads per class (`|Δlogit-diff| ≥ 0.2` — the genuine signal; Fig 3):**

| class | s | m | l | xl | 70m | 410m | 1.4b | 2.8b |
|---|---|---|---|---|---|---|---|---|
| name mover | 9 | 11 | 8 | 5 | 2 | 1 | 1 | 0 |
| s2 inhibition | 3 | 1 | 0 | 0 | 0 | 0 | 0 | n/r |
| induction | 2 | 0 | 0 | n/r | 0 | 0 | 1 | n/r |
| duplicate token | 1 | 0 | 0 | n/r | 0 | 0 | 0 | n/r |
| previous token | 1 | 0 | 0 | n/r | 0 | 0 | 0 | n/r |
| negative | 2 | 2 | 2 | 1 | 1 | 1 | 0 | 0 |

Read-off:
- **Name-mover is the only universal class with strong heads at every GPT-2 rung**
  (9/11/8/5). It also carries strong heads on Pythia rungs that half-work (70m:2,
  410m:1, 1.4b:1), and only vanishes on pythia-2.8b (which fails the task).
- **Negative name-movers** are the second-most persistent (2 across GPT-2; 1 on
  gpt2-xl and the two smallest Pythia).
- **S-inhibition** has strong single heads only in GPT-2-small (3) and weakly
  gpt2-medium (1); by gpt2-large and across all Pythia it is 0 strong heads.
- **The upstream chain (induction / duplicate / previous-token)** has strong single
  heads *essentially only in GPT-2-small*. Everywhere else these classes are carried
  (if at all) by many weak heads, not identifiable single heads.

Interpretation: with scale the IOI mechanism does not disappear — the model still does
the task — but it becomes **distributed**: the sharp single-head structure the paper
found in GPT-2-small (26 heads, 7 classes) is a GPT-2-small phenomenon, and by
medium/large only the output-side name-mover/negative heads remain individually
identifiable.

### 2. Scale fits (OLS on log₁₀ params; coefficient + R²; n small — read with care)

| relation | n | slope / decade | R² | p | verdict |
|---|---|---|---|---|---|
| **faithfulness F(C)/F(M) vs log-params — GPT-2 (F(M)>0)** | 4 | **−0.61** | **0.75** | 0.13 | falls with scale, **not significant** |
| faithfulness vs log-params — GPT-2 + pythia-1.4b | 5 | −0.28 | 0.14 | 0.54 | flattened by the one good Pythia point |
| **#circuit-heads (template) vs log-params** | 6 | **0.00** | **0.00** | 1.0 | **degenerate — fixed at 26 by construction** |
| #strong-heads (total) vs log-params (abcd rungs) | 6 | −3.65 | 0.07 | 0.62 | downward but no real fit |
| #name-mover heads (template) vs log-params | 8 | 0.00 | 0.00 | 1.0 | degenerate (fixed 11) |
| #name-mover **strong** heads vs log-params | 8 | −2.45 | 0.10 | 0.44 | weak downward, noisy |
| #S-inhibition heads (template) vs log-params | 7 | 0.00 | 0.00 | 1.0 | degenerate (fixed 4) |
| #S-inhibition **strong** heads vs log-params | 7 | −1.05 | 0.22 | 0.28 | downward, noisy |
| baseline logit-diff vs log-params — GPT-2 | 4 | +0.51 | 0.29 | 0.46 | roughly flat/mild-rise |
| baseline logit-diff vs log-params — Pythia | 4 | +0.46 | 0.25 | 0.50 | rises but stays ≤0.85 |
| baseline logit-diff vs log-params — all incl. Tier-2 | 10 | +0.50 | 0.01 | 0.75 | no size trend; generation dominates |

Honest verdicts:
- **Faithfulness falls within GPT-2** (anchor 87.8% → medium 41.5% → large 11.5%;
  gpt2-xl 28.4% is a 3-class partial), slope −0.61/decade with R²=0.75 — this
  **overturns the flat-faithfulness prior *within GPT-2***, but with n=4 and p=0.13 it is
  not statistically significant, and the drop is at least partly an artifact of the fixed
  top-k template (it does not re-select the minimal head set per model). The GPT-2-small
  anchor value is the paper-curated circuit; medium/large use auto-discovered heads.
- **Head-count-vs-scale cannot be estimated from 001** — the circuit size is a fixed
  template (26), so slope is 0 by construction and the sub-linear-growth hypothesis is
  **untestable from these artifacts** (see Gap, below). Same for the per-class template
  counts of name-mover and S-inhibition.
- Using the *strong-head* proxy instead, both #name-mover and #S-inhibition strong heads
  trend **down** with log-params (−2.45 and −1.05/decade) with low R² — consistent with
  "mechanism becomes distributed," but too noisy for a quantitative law (n≈7–8).
- **Behaviorally, size barely matters; generation matters.** Within GPT-2 logit-diff is
  ~flat-to-slightly-rising; pooled across generations there is no size trend (R²=0.01)
  because the Pythia family sits far below everything else.

### 3. Generation contrast (GPT-2 learned pos-emb, 2019 vs Pythia rotary, 2023)

| | GPT-2 family | Pythia family |
|---|---|---|
| baseline logit-diff | 3.49 / 3.58 / 4.48 / 3.79 (s/m/l/xl) | −0.57 / −0.38 / **+0.85** / −0.34 (70m/410m/1.4b/2.8b) |
| IO>S rate | 99.5 / 100 / 99.9 / 99.9 % | 37.8 / 39.6 / 71.6 / 40.7 % |

At matched scale the gap is stark: **gpt2-large (774M) logit-diff 4.48 and gpt2-xl
(1.5B) 3.79 vs pythia-1.4b (1.4B) 0.85** — the older, smaller GPT-2 architecture does IOI
cleanly where the newer, larger Pythia largely fails (prefers the repeated subject name).
The single exception, pythia-1.4b, is the *only* Pythia rung with F(M)>0, and its
**discovered-circuit faithfulness is 87.7% — essentially identical to GPT-2-small's
87.8%/88.6%**. So the discovery method itself transfers across generation/positional
scheme (rotary, NeoX) *when the behavior is present*; what differs by generation is
whether the behavior exists at all.

### 4. Behavioral-only rung (Tier-2, folded into behavior only — NOT circuit fits)

Using 001's actual Tier-2 artifacts (`tier2_*.json`, raw-HF, no circuit):
**Qwen2.5-0.5B logit-diff +5.06 (IO>S 100%)** and **SmolLM2-1.7B +4.17 (IO>S 99.2%)** —
both *exceed* GPT-2-small's 3.49. Modern small models (2024) do IOI at least as well as
GPT-2, confirming the behavior is not lost with generation *in general* — the Pythia
failure is Pythia-specific, not "post-2019-specific." These points enter only the
behavioral logit-diff trend (Fig 2), never the circuit-structure fits, per instruction.

### Updated claim, in words

> **The IOI name-mover backbone persists across GPT-2 scale (small→XL) and into the one
> Pythia model that does the task (1.4b): name-mover is the only class with strong single
> heads at every GPT-2 rung, and negative-name-mover is the second-most persistent.
> S-inhibition and the upstream induction/duplicate/previous-token chain have strong
> single heads essentially only in GPT-2-small — by GPT-2-medium/large the mechanism
> becomes distributed and the fixed-template circuit's faithfulness falls with scale
> (GPT-2 slope −0.61 per decade of params, R²=0.75, n=4, p=0.13; anchor 87.8% → medium
> 41.5% → large 11.5%), overturning the flat-faithfulness prior within GPT-2 though not at
> significance. Circuit head-count could not be tested for scaling because 001 fixed it at
> 26 by template. Generation dominates size behaviorally: the entire Pythia family (rotary,
> 2023) fails IOI (logit-diff ≤0.85, mostly negative) where every GPT-2 and every modern
> small model (Qwen2.5-0.5B +5.06, SmolLM2-1.7B +4.17) succeeds; the one working Pythia rung
> has GPT-2-small-like faithfulness (87.7% vs 87.8%), so the circuit *method* transfers
> across architecture when the behavior is present. S-inhibition/induction/duplicate/
> previous-token are GPT-2-small-specific as identifiable single heads; name-mover is
> universal.**

Every number cites a 001 artifact: baselines/faithfulness/head-lists from
`followup/001-living-update/results/per_model/*.json` and `followup_summary.json`;
anchors from `run/replication/step2/step8`.

## Deviations & limitations

1. **Head-count scaling is untestable from 001 (the biggest limitation).** 001's
   discovery uses a fixed per-class top-k template (26 heads), so #heads, #name-mover, and
   #S-inhibition are constant across models — the requested "#heads vs log-params" and the
   sub-linear-growth hypothesis cannot be evaluated. I report the degenerate fits
   (slope 0, R²=0) honestly and substitute a **strong-head** proxy for a qualitative
   trend. **A named, small gap that would fix this:** re-run 001's greedy/minimality
   search (paper's C19/C21) for 2–3 rungs (e.g. gpt2-medium, gpt2-large) to get a
   *size-varying* discovered circuit; 001 dropped this "anchor-only." I did **not** run it
   (would be new GPU work beyond "read from disk"; flag it as the gap rather than silently
   filling it).
2. **Faithfulness fits rest on ≤5 points, none significant** (GPT-2 n=4, p=0.13). The
   within-GPT-2 collapse is real in the data but partly reflects the non-adaptive template
   and the reduced path-patch N for large models (N=30 for large/xl vs N=100 for small);
   the GPT-2-small anchor is paper-curated while medium/large are auto-discovered. Do not
   over-read the −0.61 slope as a law.
3. **gpt2-xl and pythia-2.8b are partial** (3-class / 2-class; configs c,d / c,d,b not run
   in 001). Their cells are `n/r`, not 0. gpt2-xl's 28.4% faithfulness is a 3-class
   output-side circuit.
4. **Faithfulness ratios excluded where F(M)≤0.** pythia-70m/410m/2.8b fail IOI (F(M)<0),
   so F(C)/F(M) is mathematically defined but meaningless; they are omitted from
   faithfulness fits and figures (noted on Fig 1).
5. **Tier-2 set differs from the instruction.** The 002 instruction names Tier-2 as
   Qwen2.5-0.5B / Llama-3.2-3B / gemma-2-2b, but 001 actually produced Qwen2.5-0.5B and
   **SmolLM2-1.7B** (Llama/gemma were not run; 001 also notes OLMo-2 was skipped). Per
   "no new GPU jobs / read from disk," I used 001's actual Tier-2 artifacts and did not
   run the instruction's named models.
6. **Behavioral vs circuit survival are kept separate** throughout: Tier-2 enters only the
   logit-diff trend, never the structure fits.

## Artifacts

- `results/fit_results.json` — all series, both persistence matrices, every fit
  (coeff/R²/p/n), generation contrast.
- `results/fig1_faithfulness_vs_scale.png`, `results/fig2_behavioral_vs_scale.png`,
  `results/fig3_class_persistence_heatmap.png`.
- `workspace/analyze_scaling.py`, `workspace/make_figures.py`.
