# 001-living-update — Does the IOI circuit survive in today's models?

**Paper:** Wang et al. 2022, "Interpretability in the Wild: a Circuit for Indirect Object Identification
in GPT-2 small" (arXiv 2211.00593). **Replication (0.761, 21/24)** at
`/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/ioi-gpt2-circuit/run/`.

**Mechanism:** run with
`python -m veritas.cli.main followup <run_dir> --instruction-file <this file> --name living-update`
→ writes into `<run_dir>/followup/001-living-update/`. **Reuse `<run_dir>/replication/codebase/`.**

---

## Question
The paper characterizes a 26-head, 7-class IOI circuit in GPT-2 small and reports concrete
faithfulness / completeness / behavioral numbers. **Do the IOI *task behavior* and the *circuit
structure* — and those numbers — survive as we move (a) up in scale within one architecture (GPT-2
small→medium→large→XL) and (b) across generations (Pythia 70M→2.8B; and behaviorally into modern small
models)?** Produce, in one comparison table, the paper's original numbers, the on-disk replicated
numbers, and freshly computed per-model numbers.

---

## Claims this study UPDATES (with paper value, replicated value, and artifact path)
All paper/replicated numbers below are **read from disk** (do not recompute the originals):

| ID | Quantity | Paper value | Replicated value | Read original from |
|---|---|---|---|---|
| C2 | baseline IOI: logit-diff / IO>S / IO-prob | 3.56 / 99.3% / 49% | 3.49 / 99.5% / 50.2% | `run/replication/step2_baseline_result.json` |
| C3 | faithfulness F(M), F(C), F(C)/F(M) | F(M)=3.56, ~87% | F(M)=3.2645, F(C)=2.8655, **87.8%** | `run/replication/step8_completeness_result.json` |
| C4 | name-mover + neg-NM heads (h→Logits) | NM 9.6/9.9/10.0; negNM 10.7/11.10 | same | `run/replication/step3_pathpatching_result.json` |
| C9 | S-inhibition heads (h→NM queries) | 7.3/7.9/8.6/8.10 | same | `run/replication/step3_b_h_to_NMquery_heads.npy` |
| C14 | induction + duplicate heads (h→S-inh values) | ind 5.5/6.9; dup 0.1/3.0 | same | `run/replication/step3_c_h_to_SInhval_heads.npy` |
| C16 | previous-token heads (h→induction keys) | 4.11/2.2 | same | `run/replication/step3_d_h_to_INDkey_heads.npy` |
| C1 | circuit size / #classes | 26 heads, 7 classes | 26, 7 | `run/replication/codebase/easy_transformer/ioi_circuit_extraction.py` (CIRCUIT) |
| C5 | name-mover attention on IO | 0.59 | 0.584 | `run/replication/step4_namemovers_result.json` |
| C11 | S-inhibition attention on S2 | 0.51 | 0.437 | `run/replication/replication_log.json` (step 5) |

**Headline number carried across the whole ladder:** circuit **faithfulness F(C)/F(M)** (anchor GPT-2
small = **87.8%**) and **baseline mean logit-diff** (anchor = **3.49**).

## Claims this study does NOT update (and why)
- **C19, C21** (greedy incompleteness / naive-circuit faithfulness): the greedy+random completeness
  search alone took ~42 min (2499 s) for GPT-2 small (`run/replication` step 8). Laddering it blows the
  budget. Faithfulness is kept; completeness search is **anchor-only**. State this as a deliberate scope cut.
- **C6/C7/C8/C10/C12/C13/C17/C18/C20/C22** (writing-direction correlation, copy scores, signal grid,
  backup-NM, minimality, adversarial): second-order characterizations. Optionally recompute for GPT-2
  medium only if budget remains; otherwise carry anchor values forward unchanged and mark "anchor-only."
  C22 (adversarial) uses GPT-2-name-specific templates → do not port to non-GPT-2 tokenizers.
- **C15, C23, C24**: `out_of_scope` in the original plan (not shipped as runnable) — leave as-is.

---

## Model ladder (verified 2026-09-09: keyless HF API + local cache `ls`)
Storage: **only** under `/net/projects2/chai-lab-models/haokunliu/`. Set `HF_HOME=.../alignment-batch/hf-cache`,
`HF_HUB_OFFLINE=1` (all Tier-1 rungs + several Tier-2 are already cached). NEVER write under `/home`.

### Tier 1 — full circuit pipeline (vendored `easy_transformer` fork; all cached, ungated)
| HF id | params | local cache (models--…) | gated | TL(fork)-supported | tokenizer |
|---|---|---|---|---|---|
| gpt2 | 124M | `shared_models/hub/models--gpt2` | no | yes | GPT-2 (**anchor — already done**) |
| gpt2-medium | 355M | `models--gpt2-medium` | no | yes | GPT-2 (identical → names valid) |
| gpt2-large | 774M | `models--gpt2-large` | no | yes | GPT-2 (identical) |
| gpt2-xl | 1.5B | `models--gpt2-xl` | no | yes | GPT-2 (identical) |
| EleutherAI/pythia-70m | 70M | `models--EleutherAI--pythia-70m` | no | yes (neox) | NeoX (filter names) |
| EleutherAI/pythia-410m | 410M | `models--EleutherAI--pythia-410m` | no | yes (neox) | NeoX (filter names) |
| EleutherAI/pythia-1.4b | 1.4B | `models--EleutherAI--pythia-1.4b` | no | yes (neox) | NeoX (filter names) |
| EleutherAI/pythia-2.8b | 2.8B | `models--EleutherAI--pythia-2.8b` | no | yes (neox) | NeoX (filter names) |

Ungated download fallbacks (not cached): `EleutherAI/pythia-160m`, `pythia-1b`.

### Tier 2 — behavioral-baseline-only (modern arch; fork CANNOT load → raw HF `AutoModelForCausalLM`)
No circuit discovery (would need a modern `transformer_lens` port = out of budget). Only C2-style
behavioral metric, reusing the SAME IOIDataset prompt strings, scored at END over single-token IO/S.
| HF id | params | local cache | gated (HF) | note |
|---|---|---|---|---|
| Qwen/Qwen2.5-0.5B | 494M | `models--Qwen--Qwen2.5-0.5B` | **no** | primary modern rung (cached+ungated) |
| meta-llama/Llama-3.2-3B | 3.2B | `models--meta-llama--Llama-3.2-3B` | manual | cached → load **offline**, gate not hit |
| google/gemma-2-2b-it | 2.6B | `models--google--gemma-2-2b-it` | manual | cached → load offline |
| allenai/OLMo-2-0425-1B | 1.48B | (download) | no | ungated fallback |
| HuggingFaceTB/SmolLM2-1.7B | 1.7B | (download) | no | ungated fallback |

Do **not** probe gated repos with a token. Tier-2 is optional colour; Tier-1 is the required evidence.

---

## Exact reuse of `run/replication/codebase/` (scripts, args, model swap)

Set up a fresh venv under the followup dir with **uv** (do NOT reuse/modify the replication `.venv`):
```
cd <run_dir>/followup/001-living-update
uv venv && source .venv/bin/activate
uv pip install -e <run_dir>/replication/codebase
uv pip install -r <run_dir>/replication/codebase/requirements.txt   # torch 2.2.2, transformers 4.30.2, numpy<2
export HF_HOME=/net/projects2/chai-lab-models/haokunliu/alignment-batch/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
```
Every entry script asserts `torch.cuda.device_count()==1` → keep exactly one device visible.

**Copy (do not edit in place) the replication step scripts as the per-model driver templates:**
`step2_baseline.py`, `step3_pathpatching.py`, `step8_completeness.py` from `<run_dir>/replication/`.
The replication dir is READ-ONLY; write copies into `followup/001-living-update/`.

**Model swap (the ONE change that generalizes the driver):**
- `step2_baseline.py:10` and `step3_pathpatching.py:20`: `EasyTransformer.from_pretrained("gpt2")`
  → parameterize as `from_pretrained(MODEL_ID)`. GPT-2 medium/large/xl are already valid names.
- Pythia registration (once, before loading):
  ```python
  EasyTransformer.VALID_PRETRAINED_MODEL_NAMES |= {
      "EleutherAI/pythia-70m","EleutherAI/pythia-410m",
      "EleutherAI/pythia-1.4b","EleutherAI/pythia-2.8b"}
  ```
  Then assert `m.cfg.positional_embedding_type=="rotary"` after load.
- **Parameterize the two GPT-2-small hardcodes** in `step3_pathpatching.py`:
  1. config a receiver `blocks.11.hook_resid_post` → `blocks.{m.cfg.n_layers-1}.hook_resid_post`.
  2. `name_movers`, `induction`, `circuit["s2 inhibition"]` (receivers for configs b/c/d) are GPT-2-small
     heads — **discover them per model from config a first**, then point b/c/d at the discovered heads.

**Reuse entry points (confirmed on disk):**
- Baseline (C2): `from easy_transformer.ioi_utils import logit_diff, probs`; build
  `IOIDataset(prompt_type="mixed", N=N, tokenizer=m.tokenizer, prepend_bos=False)`;
  `logit_diff(m, d, all=True).mean()`, `(ld>0).float().mean()`, `probs(m, d)`.
- Discovery (C4/C9/C14/C16): `step3_pathpatching.py` `run_sweep(receiver_hooks, position, title, tag)` —
  the generic path-patch sweep (xorig=pIOI, xnew=pABC, % change in logit-diff over all senders).
- Faithfulness (C3): `from easy_transformer.ioi_circuit_extraction import do_circuit_extraction,
  get_heads_circuit`. F(M) = `logit_diff` on the full model over pIOI; F(C) = `logit_diff` after
  `do_circuit_extraction(heads_to_keep=<per-model circuit>, ioi_dataset=pIOI, mean_dataset=<pABC>,
  model=m, metric=logit_diff)`. Report F(C)/F(M).

**Single-token-name filter (MANDATORY for every non-GPT-2 model):** before building IOIDataset,
```python
from easy_transformer.ioi_dataset import NAMES
names1 = [n for n in NAMES if len(m.tokenizer(" "+n).input_ids)==1]
```
Pass `names1` into IOIDataset (its `nb_names`/name handling) and **report `len(names1)` per model**. If
`len(names1) < 15`, mark that model's IOI metric **"not attempted"** (mis-indexed metric is worse than a
hole). GPT-2 medium/large/xl share GPT-2's tokenizer → filter is a no-op (all 99 survive).

---

## Per-model computation plan (the ladder)

For **each Tier-1 model** (small is already done — read its numbers from disk, do not rerun):
1. **Baseline (C2):** logit-diff / IO>S / IO-prob at **N≤2000** (batched; seconds). Record.
2. **Discovery — config a (C4):** path-patch h→Logits @ END, **N=100**. Name movers = top negative-effect
   heads; neg-NM = top positive. Record head list + %effects.
3. **Discovery — config b (C9):** path-patch h→(discovered NM) queries @ END, N=100 → S-inhibition heads.
4. **Discovery — configs c,d (C14/C16):** ONLY for n_layers≤24 (gpt2-medium, pythia-70m/410m/1.4b) or if
   wall-clock allows; for gpt2-xl & pythia-2.8b run **a (+b if time) only**.
5. **Per-model circuit assembly:** top-k heads per class matching the paper's class counts (name mover,
   negative, s2 inhibition, induction, duplicate token, previous token) from steps 2–4. Label this
   "discovered circuit" (NOT the paper's hand-curated 26-head set).
6. **Faithfulness (C3):** F(M), F(C), F(C)/F(M) via `do_circuit_extraction` keep-eval at N=100.
7. Optional (C5/C11 attention read-outs) if budget remains.

For **each Tier-2 model:** steps 1 + name-filter only (behavioral C2 analog via raw HF). No discovery,
no faithfulness. Clearly labeled behavioral-only.

**Bounding (hard):** N=100 for all path patching; skip greedy/random completeness search entirely;
faithfulness eval only. Target **≤3.5 GPU-h**, hard cap **≤8 GPU-h**, one A40/A100. Run everything
**synchronously in this session**.

---

## The over-time / scale figure (under `results/`)
Produce with matplotlib (headless: `import matplotlib; matplotlib.use("Agg")`; write PNG to
`followup/001-living-update/results/`):
- **fig 1 `results/faithfulness_vs_scale.png`** — x = model params (log scale), y = faithfulness F(C)/F(M)
  and (twin or second series) baseline logit-diff; one marker per model, GPT-2 family and Pythia family in
  distinct series; the GPT-2-small anchor (87.8%) annotated. Horizontal line at the paper's 87%.
- **fig 2 `results/headcount_vs_scale.png`** — x = params, y = #heads per class in the discovered circuit
  (stacked or grouped bars: name mover / S-inhibition / induction / duplicate / previous / negative).
Keep raw per-model numbers in `followup_summary.json` so 002 can refit without re-running.

---

## Environment constraints
- 1 GPU (A40 48GB / A100 80GB); saturated SLURM. Small models fit; run sequentially.
- Storage under `/net/projects2/chai-lab-models/haokunliu/` only; `HF_HOME` = `.../alignment-batch/hf-cache`;
  `HF_HUB_OFFLINE=1`. **Never** `/home`.
- Python via **uv**, own `.venv` under the followup dir. Do not modify `run/replication/`.
- Reuse cached weights; if a Tier-1 Pythia rung fails to load via the fork, fall back to Tier-2
  behavioral-only for Pythia and log the load failure as an **environmental** finding (the paper never
  claimed Pythia — not a paper flaw).

---

## Deliverables (write into `followup/001-living-update/`)
1. **`report.md`** — a single **comparison table** with columns: *claim / quantity | paper original |
   artifact file (where the original was read) | GPT-2 small (repl) | gpt2-medium | gpt2-large | gpt2-xl |
   pythia-70m | pythia-410m | pythia-1.4b | pythia-2.8b | Tier-2 (behavioral)*. One row per updated claim
   (C2 baseline, C3 faithfulness, C1 head counts, C4/C9/C14/C16 discovered head sets, C5/C11 if computed).
   Plus prose: which head classes persist vs vanish with scale/generation, the single-token-name survivor
   counts per model, and every hole / fallback / hand-coded value disclosed explicitly.
2. **`followup_summary.json`** — machine-readable per-model raw numbers (baseline, faithfulness, discovered
   circuit head lists per class, name-survivor counts, GPU-seconds per model). This is 002's input.
3. **`result_card.json`** — schema `sai.followup.result_card/v1`:
   - `headline`: e.g. "IOI-circuit faithfulness across scale: GPT-2 small 87.8% → gpt2-xl X% → pythia-2.8b Y%"
     (fill with **this session's** numbers).
   - `status`: `ok` / `partial` / `blocked`.
   - 1–5 `metrics`, each with `baseline` + `provenance` (baseline = the on-disk replicated value with its
     artifact path; provenance = this session's script + args). Suggested: faithfulness F(C)/F(M);
     baseline logit-diff; #name-mover heads; #S-inhibition heads; name-survivor count.
   - ≤2 tables (the comparison table; the discovered-circuit-by-class table).
   - 0–3 figures under `results/` (the two above).
   - `notes`: scope cuts (completeness search dropped), Tier-2 = behavioral-only, Pythia tokenizer caveat.

---

## Anti-goals (MANDATORY)
- **NEVER background a long job and end the turn to wait.** Run every sweep synchronously; the budget is
  sized so this is possible on one GPU.
- **No token budgets / no "I'll continue later."** Finish the ladder in-session or reduce the ladder
  (drop the most expensive rung) and say so.
- **Disclose every hole, fallback, and hand-coded value.** A model that failed to load, a name filter that
  left too few names, a config skipped for budget — all stated explicitly in `report.md`.
- **Every new number comes from THIS session's execution.** Never copy a "new" number from the paper or a
  guess.
- **Every original/paper number is READ from the run-dir artifacts** named in the tables above, not
  recomputed and not remembered.
- Do not overclaim: Tier-2 shows *behavioral* survival, not *circuit* survival; per-model faithfulness is
  faithfulness of a *discovered* circuit, not the paper's hand-curated one.