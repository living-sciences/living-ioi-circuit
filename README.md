# Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small

This is a living-paper repository built around one paper:

> Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob
> Steinhardt. "Interpretability in the Wild: a Circuit for Indirect Object
> Identification in GPT-2 Small." International Conference on Learning
> Representations (ICLR) 2023.
> arXiv:2211.00593. DOI: 10.48550/arXiv.2211.00593.
> Official code: https://github.com/redwoodresearch/Easy-Transformer

The paper reverse-engineers how GPT-2 small (124M) solves the Indirect Object
Identification task, where a sentence like "When Mary and John went to the
store, John gave a drink to" should be completed with " Mary". The authors
identify a circuit of 26 attention heads in 7 functional classes and validate
it against three criteria: faithfulness, completeness, and minimality.

## What a living-paper repository is

A living paper starts from a careful replication of a published result and then
keeps going. Instead of freezing at "we reproduced the numbers," it grows a
small series of follow-up studies that poke at the original claims: do they
hold at larger scale, across model generations, or as the field moves on? Each
study lives beside the replication it builds on, with its own instructions,
code, and results, so the whole thing reads as one evolving record rather than
a single snapshot. The living page for this paper (see below) is where the
narrative is told for a general reader; this repository is the working code and
data behind it.

## What this repo contains

Everything lives under `run/`.

**Replication** (`run/replication/`): a full ten-step reproduction of the
paper's headline findings on GPT-2 small. The baseline IOI mean logit
difference comes out at 3.49 over 100,000 examples (the paper reports 3.56),
the model prefers the indirect object over the subject 99.5% of the time, and
the discovered circuit recovers about 87.8% of the model's logit difference.
The 26-head, 7-class circuit, the name-mover and backup-name-mover behaviour,
the S-inhibition mechanism, completeness, minimality, and the adversarial
examples all reproduce. Code is the authors' EasyTransformer drop (vendored
under `run/replication/codebase/`, with our changes recorded in
`codebase.diff`); per-step scripts, result JSON, small result tensors, and
figures sit alongside it, with `replication_log.json` and
`evidence_summary.json` documenting how the run went.

**Study 001 — scale and generation ladder** (`run/followup/001-living-update/`):
IOI behaviour survives across GPT-2 small through XL and reappears, stronger, in
modern small models (Qwen2.5-0.5B logit difference +5.06), but not in the Pythia
generation, which mostly prefers the wrong name; meanwhile the mechanically
discovered fixed-size circuit's faithfulness collapses as GPT-2 grows, from
87.8% at the anchor to 41.5% at medium and 11.5% at large.

**Study 002 — what scales, size or generation** (`run/followup/002-theory-update/`):
the name-mover backbone is the one circuit class that stays strong across every
GPT-2 rung and into the working Pythia model, but the fixed circuit's
faithfulness falls with GPT-2 size (slope about -0.61 per decade, R-squared 0.75,
n=4, not significant) and generation rather than size decides behaviour: the
whole Pythia family fails IOI while GPT-2 and modern small models succeed.

**Study 003 — does it still hold in 2026** (`run/followup/003-living-update/`):
IOI behaviour still holds, and holds more strongly, in the newest models, with
Qwen3.5-9B-Base at logit difference +5.46 (IO over S 99.0%) and gemma-4-12B at
+5.71 (IO over S 100%), both roughly 1.6 times the GPT-2 small replication
anchor of +3.49. These are behavioural tests only; the 2026 architectures could
not be wrapped for circuit discovery.

Each study folder keeps its original `instruction.md`, the `workspace/` code,
the `results/` (figures, JSON, CSV, small tensors), the `report.md`, and the
machine-readable `result_card.json` and `followup_summary.json`.

## Living page

The reader-facing living page for this paper is at:

https://livingscience.ai/safety/living-ioi-circuit

## A note on the original authors' terms

The vendored code under `run/replication/codebase/` is the original
EasyTransformer release by the paper's authors and is covered by its own
`LICENSE` (MIT). Please respect the original authors' license and citation
terms when reusing it. The replication and follow-up code and results in this
repository are our own additions built on top of that work.

## Large and omitted files

No model weights or activation files are stored in this repository, and nothing
large was carried over. See `LARGE_FILES_OMITTED.md` for the full list of what
is deliberately not included and how to obtain or regenerate it.
