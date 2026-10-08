# 003-living-update — Does the IOI behaviour survive into the 2026 models?

**Paper:** Wang et al. 2022, *Interpretability in the Wild: a Circuit for Indirect Object
Identification in GPT-2 small* (arXiv 2211.00593).
**This tick:** a 2026 model-refresh extending the 001 living-update. The 001 ladder stopped at 2024
models; here I ADD the newest 2026 flagships — **Qwen3.5-9B (base + instruct, Mar 2026)** and
**gemma-4-12B (base)** — as behavioural IOI points, reuse every prior per-model result unchanged,
re-aggregate old+new, and extend the over-time figure to 2026.

## Question

The paper shows GPT-2 small solves Indirect Object Identification (IOI): given "…Mary and Brandon …
Brandon gave a computer to __", it strongly prefers the indirect object (Mary) over the repeated
subject (Brandon), with mean logit-diff **3.56** and IO>S **99.3 %** (replication on disk:
**3.490 / 99.51 %**, `run/replication/step2_baseline_result.json`). The 001 living-update laddered this
behaviour across GPT-2 scale, the Pythia generation, and two 2024 "modern" small models. **Does the IOI
task behaviour still hold in the newest 2026 models?** For architectures the interpretability tooling
cannot wrap, the answerable question is the **behavioural** one (Tier-2 of 001): logit-diff and %IO>S.

## Approach

**Reuse (no recompute).** I copied 001's `results/per_model/*` (all 8 Tier-1 circuit results and both
2024 Tier-2 behavioural results) and its scripts/prompt set into this study, and treated every 001 model
as DONE. Every "original"/prior number below is read from those on-disk artifacts.

**New computation (2026 models only).** The 2026 flagships are multimodal-architecture
(`Qwen3_5ForConditionalGeneration`, `Gemma4UnifiedForConditionalGeneration`) and cannot be loaded by the
vendored `easy_transformer` fork / TransformerLens, nor wrapped by nnsight. As the instruction directs, I
loaded them with **plain transformers** via `AutoModelForImageTextToText` (`dtype=bfloat16`,
`device_map="auto"`) and ran the **behavioural IOI test only** — the exact Tier-2 protocol from 001:
the same 200 IOI prompt strings (`data/ioi_prompts.json`), the mandatory single-token-name filter in each
model's own tokenizer, and END-scoring of the next-token logit for `" IO"` minus `" S"`
(`workspace/tier2_behavioral_2026.py`, a one-loader-line edit of 001's `tier2_behavioral.py`).

**Smoke test first.** Before the full run I confirmed Qwen3.5-9B-Base loads, does one forward
(`logits [1,T,248320]`) and one activation read (`output_hidden_states`, len 33 = 32 layers + 1). The
text decoder is `m.model.language_model` (a `Qwen3_5TextModel`, 32 layers) — **not** `m.language_model`
as the instruction's sketch had it (transformers 5.17.0 nests it under `.model`); the logits path
`m(input_ids=ids).logits` works exactly as sketched, and that is all the behavioural test needs.

**Environment.** The reused 001 Tier-2 venv (transformers 4.46.3) does not know `model_type qwen3_5`, so
I built a fresh `.venv-2026` (torch 2.14.0 / transformers 5.17.0, which registers both `qwen3_5` and
`gemma4_unified`). All weights loaded offline from the local cache; nothing written under `/home`.

## Results

### The new 2026 numbers (this session's execution)

| Model | Params | arch (loader) | single-tok names | logit-diff (IO−S) | IO>S | IO prob | N |
|---|---|---|---|---|---|---|---|
| **Qwen3.5-9B-Base** | 9B | Qwen3_5ForConditionalGeneration | 97/97 | **+5.464** | 99.0 % | 34.9 % | 200 |
| **Qwen3.5-9B (instruct)** | 9B | Qwen3_5ForConditionalGeneration | 97/97 | **+5.840** | 99.5 % | 37.7 % | 200 |
| **gemma-4-12B (base)** | 12B | Gemma4UnifiedForConditionalGeneration | 97/97 | **+5.712** | 100.0 % | 37.8 % | 200 |

All read from `results/per_model/tier2_{Qwen__Qwen3.5-9B-Base,Qwen__Qwen3.5-9B,google__gemma-4-12B}.json`.

### Full ladder — behavioural IOI over generations (originals reused from disk)

| Model | ~Year | Params | Baseline logit-diff | IO>S | Does IOI? | Source |
|---|---|---|---|---|---|---|
| GPT-2 small (paper) | 2019 | 124M | 3.56 | 99.3 % | yes | paper |
| **GPT-2 small (replication anchor)** | 2019 | 124M | **3.490** | **99.51 %** | yes | `run/replication/step2_baseline_result.json` |
| gpt2-medium | 2019 | 355M | 3.583 | 100 % | yes | 001 `per_model/gpt2-medium.json` |
| gpt2-large | 2019 | 774M | 4.481 | 99.95 % | yes | 001 `per_model/gpt2-large.json` |
| gpt2-xl | 2019 | 1.5B | 3.789 | 99.9 % | yes | 001 `per_model/gpt2-xl.json` |
| pythia-70m | 2023 | 70M | −0.572 | 37.8 % | **no** | 001 `per_model/…pythia-70m.json` |
| pythia-410m | 2023 | 410M | −0.380 | 39.6 % | **no** | 001 |
| pythia-1.4b | 2023 | 1.4B | 0.852 | 71.6 % | weak | 001 |
| pythia-2.8b | 2023 | 2.8B | −0.336 | 40.6 % | **no** | 001 |
| Qwen2.5-0.5B | 2024 | 494M | 5.064 | 100 % | yes | 001 `per_model/tier2_Qwen__Qwen2.5-0.5B.json` |
| SmolLM2-1.7B | 2024 | 1.7B | 4.172 | 99.2 % | yes | 001 `per_model/tier2_…SmolLM2-1.7B.json` |
| **Qwen3.5-9B-Base** | **2026** | **9B** | **+5.464** | **99.0 %** | **yes** | **this session** |
| **Qwen3.5-9B (instruct)** | **2026** | **9B** | **+5.840** | **99.5 %** | **yes** | **this session** |
| **gemma-4-12B (base)** | **2026** | **12B** | **+5.712** | **100 %** | **yes** | **this session** |

### Headline

**The paper's IOI finding still holds — and is behaviourally stronger — in the newest 2026 models.**
Qwen3.5-9B (base +5.46 / instruct +5.84) and gemma-4-12B (+5.71) all show a large positive IO−S
logit-diff and ≥99 % IO>S, i.e. ~1.6× the GPT-2-small replication anchor (+3.49). The 2026 models sit at
the very top of the IOI-strength range, alongside the 2024 modern models (Qwen2.5-0.5B +5.06) and well
above every GPT-2 rung. The only generation that ever *failed* IOI on this ladder remains the 2023 Pythia
family (three of four rungs prefer the repeated name), which the paper never claimed — so the failure is a
property of that pre-training recipe, not a break in the paper's claim.

### Figures (under `results/`)

- **`behavioral_over_time.png`** — the over-time figure, now extended to 2026: baseline IOI logit-diff vs
  approximate release year, GPT-2 (2019) → Pythia (2023) → modern-2024 → **2026 (stars)**. Shows IOI
  behaviour holding and strengthening into 2026, with the Pythia dip at/below zero.
- **`behavioral_vs_scale.png`** — the same points vs parameter count (log-x), 2026 flagships added.
- **`faithfulness_vs_scale.png`** — the Tier-1 discovered-circuit faithfulness ladder from 001,
  regenerated unchanged for self-containedness (the 2026 models are behavioural-only and contribute no
  faithfulness point — see Limitations).

## Deviations & limitations

- **Behavioural-only for the 2026 models (by design).** No circuit discovery or faithfulness (C3/C4/C9/
  C14/C16) is computed for Qwen3.5-9B or gemma-4-12B: the fork/TransformerLens cannot load `qwen3_5` /
  `gemma4_unified`, and nnsight cannot wrap them. This is exactly the 001 Tier-2 limitation, extended to
  2026. The result therefore shows **behavioural** survival of IOI, **not circuit** survival — the newest
  models could in principle solve IOI by a different internal mechanism; testing that would need a modern
  TransformerLens port (out of scope for this tick).
- **Loader attribute path.** The instruction sketched `dec = m.language_model`; in transformers 5.17.0
  the text decoder is `m.model.language_model`. Adjusted after inspecting the module tree; the behavioural
  test only uses `m(input_ids=ids).logits`, which is unaffected.
- **New venv required.** The reused 001 Tier-2 venv (transformers 4.46.3) lacks `qwen3_5`; I built a fresh
  `.venv-2026` (transformers 5.17.0 / torch 2.14.0). Weights loaded offline from the local cache; no
  `/home` writes.
- **gemma-4-12B included.** It was fully downloaded (23 GB) and hookable, so it is a genuine second 2026
  point (base variant, matching the ladder's base-model convention). Both Qwen3.5-9B base and instruct
  were run; the base variant is the ladder point, the instruct is reported as colour.
- **Approximate release years** are used only for the over-time x-axis ordering (Qwen3.5 placed at 2026.2
  per the instruction, gemma-4 at ~2026.6), never for any quantitative claim.
- **Instruct vs base.** The 2024 Tier-2 points and the GPT-2/Pythia ladder are base models; for an
  apples-to-apples generational comparison the ladder point for 2026 is **Qwen3.5-9B-Base**. Instruct
  scores marginally higher (+5.84 vs +5.46), as expected.

## Reproduce

```
cd run/followup/003-living-update/workspace
source .venv-2026/bin/activate            # transformers 5.17.0 / torch 2.14.0
export HF_HOME=/net/projects2/chai-lab-models/haokunliu/alignment-batch/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
python tier2_behavioral_2026.py <Qwen3.5-9B-Base path> Qwen/Qwen3.5-9B-Base
python tier2_behavioral_2026.py <Qwen3.5-9B path>      Qwen/Qwen3.5-9B
python tier2_behavioral_2026.py <gemma-4-12B path>     google/gemma-4-12B
python aggregate_and_figures.py           # re-aggregate old+new; extend over-time figure
```
Per-model raw numbers are in `results/per_model/tier2_*.json`; the old+new roll-up is
`results/aggregate_all.json`; machine-readable summary in `followup_summary.json`.
