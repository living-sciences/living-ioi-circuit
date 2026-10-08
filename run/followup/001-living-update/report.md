# 001-living-update — Does the IOI circuit survive in today's models?

**Paper:** Wang et al. 2022, *Interpretability in the Wild: a Circuit for Indirect Object
Identification in GPT-2 small* (arXiv 2211.00593).
**Replication anchor** (read from disk, not recomputed): baseline mean logit-diff **3.490**,
circuit faithfulness **F(C)/F(M) = 87.8 %** (`run/replication/step2_baseline_result.json`,
`run/replication/step8_completeness_result.json`).

## Question

The paper characterises a 26-head, 6-class circuit for Indirect Object Identification (IOI) in
GPT-2 small and reports concrete behavioural and faithfulness numbers. Does the IOI **task
behaviour** and the **circuit structure** — and those numbers — survive **(a) up in scale within one
architecture** (GPT-2 small→medium→large→XL) and **(b) across generations** (Pythia 70M→2.8B, and
behaviourally into modern small models)? I recompute per-model baseline (C2), path-patch circuit
discovery (C4/C9/C14/C16) and circuit faithfulness (C3) fresh, and place them beside the paper's and
the on-disk replicated numbers.

## Approach

I reused the replication's vendored `easy_transformer` fork verbatim, in a **fresh `uv` venv** under
the follow-up dir (editable-install of `run/replication/codebase`, torch 2.2.2 / transformers 4.30.2 /
numpy<2). I copied the replication step scripts into a single parameterised driver
(`workspace/ladder_tier1.py`) that applies **one generalisation**: the model id is swapped, the config-a
receiver is `blocks.{n_layers-1}.hook_resid_post`, and the config b/c/d receivers are **discovered per
model from the previous config** (not the GPT-2-small hard-coded heads). Pythia ids were registered on
`VALID_PRETRAINED_MODEL_NAMES`; the fork already maps `"pythia"→neox` (rotary) and reads its config
generically, so all four Pythia rungs loaded. For every non-GPT-2 model I applied the mandatory
**single-token-name filter** (monkeypatching the module-global `NAMES`) and report the survivor count.

Per Tier-1 model the driver computes: **(1)** baseline logit-diff / IO>S / IO-prob at N=2000 (batched);
**(2–5)** the generic path-patch sweep for configs a (h→Logits), b (h→NM queries), c (h→S-inh values),
d (h→induction keys) at N≤100, selecting the top-k heads per class **matching the paper's class counts**
to assemble a *discovered* circuit; **(6)** faithfulness F(M), F(C), F(C)/F(M) via
`do_circuit_extraction` keep-eval on that circuit. **Tier-2** modern models (Qwen2.5-0.5B, SmolLM2-1.7B)
get a behavioural-only IOI score via raw HF `AutoModelForCausalLM` in a separate modern-`transformers`
venv, reusing the same IOI prompt strings scored at END over single-token IO/S.

**Harness validation.** Before trusting any new faithfulness number I confirmed my extraction harness
reproduces the on-disk anchor **exactly**: the paper's hand-curated circuit through this pipeline gives
**F(C)/F(M) = 0.8778** (= disk 87.8 %; `workspace/validate_anchor.py`). On GPT-2 small the *discovered*
circuit independently recovers the paper's heads (name movers 9.9/9.6/10.0, negative NMs 10.7/11.10,
S-inhibition 7.3/7.9/8.6/8.10, induction 5.5/5.9/6.9, previous-token 4.11/2.2) and gives **88.6 %**.

## Results

### Main comparison table

Original columns are **read from disk**; every "followup" number is from **this session's execution**.
Faithfulness is of the *discovered* circuit (6-class unless noted), N in parentheses.

| Quantity | Paper | Replicated (disk) | gpt2 (repl→new) | gpt2-medium | gpt2-large | gpt2-xl | pythia-70m | pythia-410m | pythia-1.4b | pythia-2.8b |
|---|---|---|---|---|---|---|---|---|---|---|
| **Params** | — | 124M | 124M | 355M | 774M | 1.5B | 70M | 410M | 1.4B | 2.8B |
| **C2 baseline logit-diff** | 3.56 | 3.490 | 3.495 | 3.583 | **4.481** | 3.789 | **−0.572** | **−0.380** | 0.852 | **−0.336** |
| **C2 IO>S rate** | 99.3% | 99.5% | 99.5% | 100% | 99.95% | 99.9% | 37.8% | 39.6% | 71.6% | 40.6% |
| **C2 IO prob** | 49% | 50.2% | 49.6% | 44.4% | 43.3% | 39.0% | 0.8% | 2.2% | 3.2% | 3.6% |
| **C3 F(M)** | 3.56 | 3.2645 | 3.502 | 3.391 | 4.435 | 3.940 | −0.704 | −0.304 | 0.688 | −0.574 |
| **C3 F(C)** | — | 2.8655 | 3.104 | 1.406 | 0.509 | 1.120 | −0.722 | −0.664 | 0.603 | −1.306 |
| **C3 F(C)/F(M)** | ~87% | **87.8%** | **88.6%**(100) | **41.5%**(100) | **11.5%**(30) | 28.4%†(30) | n/a‡ | n/a‡ | **87.7%**(50) | n/a‡ |
| **Single-token names** | 99 | 99 | 99 | 99 | 99 | 99 | 88 | 88 | 88 | 88 |
| **circuit unique heads** | 26 | 26 | 23 | 25 | 25 | 15† | 21 | 26 | 25 | 13§ |

† **gpt2-xl faithfulness is a 3-class output-side circuit** (name mover + negative + S-inhibition, 15
heads): configs c,d were dropped for budget (config b alone took 38 min at N=30). Not comparable to the
6-class rungs. § pythia-2.8b was config-a-only (2-class, 13 heads). ‡ **n/a** = the model does **not**
do the IOI task (F(M) ≤ 0, i.e. it prefers the repeated subject name), so a faithfulness *ratio* is
mathematically defined but meaningless — reported as not-interpretable rather than as a number.

*(Unique-head counts fall below 26 because a head can be top-k in two classes; the `heads_to_keep` dict
de-duplicates by (layer,head).)*

### Tier-2 — behavioural survival in modern small models (raw HF, no circuit)

| Model | Params | logit-diff | IO>S | IO prob | single-token names | prompts scored |
|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-0.5B | 494M | **+5.064** | 100% | 45.5% | 97/97 | 200 |
| HuggingFaceTB/SmolLM2-1.7B | 1.7B | **+4.172** | 99.2% | 19.7% | 79/97 | 132 |

### What persists vs. what vanishes

**1. IOI *behaviour* is a GPT-2 / modern-model property, not a Pythia property** (fig 3,
`behavioral_vs_scale.png`). Every GPT-2 rung solves IOI strongly (logit-diff 3.5–4.5, IO>S ≥ 99.9 %),
and the two **modern** small models solve it *even better* (Qwen2.5-0.5B **+5.06**, SmolLM2-1.7B
**+4.17**). But the **entire Pythia family fails**: pythia-70m/410m/2.8b have **negative** logit-diff
(they prefer the repeated name), and only pythia-1.4b is weakly positive (+0.85, IO>S 72 %). So IOI
behaviour did *not* cleanly survive into the 2023 Pythia generation, yet reappears — stronger — in
2024-era models. This points at pre-training data/recipe rather than scale or architecture as the driver
(the paper never claimed Pythia; this is not a flaw in the paper).

**2. Faithfulness of a mechanically-discovered fixed-size circuit collapses with GPT-2 scale**
(fig 1, `faithfulness_vs_scale.png`). Within GPT-2 the discovered 6-class circuit goes
**small 88.6 % → medium 41.5 % → large 11.5 %** — a monotone decay, even though the *model* keeps doing
the task perfectly. The task is still there; the *low-rank hand-sized circuit* stops capturing it.

**3. With scale the computation becomes more distributed** (fig 2, `headcount_vs_scale.png`).
Counting only discovered heads whose ablation actually moves the logit-diff by ≥0.2 (a scale-stable raw
measure): GPT-2 small has strong heads across *all six* classes (9 name-mover, 2 negative, 3
S-inhibition, 2 induction, 1 duplicate, 1 previous). By gpt2-medium/large the strong **name-mover +
negative** output heads remain (11+2, 8+2) but the strong single **S-inhibition / induction / duplicate
/ previous** upstream heads **disappear** — the value-routing chain spreads across many weak heads. The
**name-mover class is the most robust** structural signature and shifts to later absolute layers with
depth (small 9.9 → medium 19.1 → large 20.14 → XL 22.6).

**4. Where the model does the task, the discovered circuit can still be faithful across a
generation gap.** pythia-1.4b — the one Pythia rung that (weakly) does IOI — yields a discovered 6-class
circuit with **F(C)/F(M) = 87.7 %**, essentially the GPT-2-small anchor value, showing the *method*
transfers to a rotary/NeoX model when the behaviour is present.

## Deviations & limitations

- **Cache miss.** The instruction stated Tier-1 weights were cached; only `gpt2` was present under
  `HF_HOME`. I downloaded gpt2-medium/large/xl and all Pythia rungs fresh (HF online for those loads
  only). Nothing was ever written under `/home`; all weights and venvs live under
  `/net/projects2/chai-lab-models/haokunliu/`.
- **gpt2-xl full circuit not completed** (budget). config b alone took 38 min (N=30, 1200 heads);
  c,d were dropped. gpt2-xl faithfulness (28.4 %) is a **3-class output-side** circuit and is flagged
  everywhere as partial — it is **not** on the same footing as the 6-class rungs.
- **Adaptive N for large models.** Per-head qkv-input caching OOMed at N=100 on pythia-1.4b (48 GB
  A40). I disabled qkv-input caching for config a, and reduced the path-patch N to 50 (pythia-1.4b) /
  30 (gpt2-large, gpt2-xl, pythia-2.8b). Baseline (C2) is always N=2000. Faithfulness numbers at N=30
  are noisier Monte-Carlo estimates than N=100.
- **Completeness / greedy search (C19/C21) dropped entirely**, as the instruction directs (the anchor
  greedy+random search alone took 42 min). Faithfulness kept; completeness is anchor-only.
- **induction vs duplicate split** inside config c is a **layer-depth heuristic** (earliest layers →
  duplicate token). Both classes read the S2 position, so the split does not affect any faithfulness
  number — only the fig-2 class labelling.
- **Name filter** applied by monkeypatching the module-global `NAMES` (IOIDataset has no `names=` arg).
  Pythia/NeoX keeps **88/99** names (≥15 for every model, so none was marked "not attempted"); GPT-2
  models keep all 99 (tokenizer identical). Tier-2: Qwen 97/97, SmolLM2 79/97.
- **Tier-2 is behavioural-only** (no circuit — the fork cannot load these architectures). It shows
  behavioural survival, **not** circuit survival. OLMo-2-0425-1B could not load (arch `olmo2` unknown to
  transformers 4.46.3) → environmental skip; gated repos were not probed with a token.
- **Discovered ≠ curated.** Every per-model faithfulness is faithfulness of a *mechanically discovered*
  top-k circuit, not the paper's hand-curated 26-head set. The two coincide on GPT-2 small (88.6 % vs
  87.8 %), which is what licenses the cross-model comparison.

## Reproduce

```
cd run/followup/001-living-update/workspace
source .venv/bin/activate         # fork venv (torch2.2.2/transformers4.30.2)
export HF_HOME=.../alignment-batch/hf-cache CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python validate_anchor.py                                   # harness == disk 87.78%
python ladder_tier1.py gpt2 abcd                            # any Tier-1 id; configs subset of abcd
python tier2_behavioral.py Qwen/Qwen2.5-0.5B                # (.venv-tier2, transformers 4.46.3)
python make_figures.py ; python build_summary.py
```
Per-model raw numbers, discovered head lists and the npy discovery matrices are under
`results/per_model/`; machine-readable roll-up in `followup_summary.json`.
