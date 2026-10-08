# Large and omitted files

This repository is meant to be light. The policy is to omit every file that is
45 MB or larger, and to omit every model weight or activation artifact
regardless of size (`*.safetensors`, model `*.bin`, weight or activation
`*.pt`/`*.pth`, `*.ckpt`, `*.gguf`, `*.h5`, large arrays, cached Hugging Face
snapshots).

## Files omitted for size

None. After cleanup, no file in the kept trees (`run/replication/` and
`run/followup/`) reaches 45 MB. The largest kept files are figures (PNG, under
110 KB each) and the vendored source. The largest kept tensors are the
completeness greedy-search result sets in
`run/replication/codebase/pts/` and `run/replication/pts/` (all small, the
biggest about 24 KB), which are experiment outputs rather than model weights,
so they are kept.

## Model weights (never stored here, obtain from the source)

The experiments load pretrained models, but no weights are vendored. All of
them are public on the Hugging Face Hub and are downloaded on first use.

- GPT-2 small (`gpt2`, 124M). This is the paper's model. The replication loads
  it through `EasyTransformer.from_pretrained("gpt2")`, which pulls from the
  Hugging Face Hub (`openai-community/gpt2`). The weights are small
  (`pytorch_model.bin` about 548 MB) but are intentionally not included; they
  come straight from the Hub.
- The scale ladder in Study 001/002: `gpt2-medium`, `gpt2-large`, `gpt2-xl`,
  and the Pythia family (`EleutherAI/pythia-70m`, `-410m`, `-1.4b`, `-2.8b`),
  plus the modern small models `Qwen/Qwen2.5-0.5B` and
  `HuggingFaceTB/SmolLM2-1.7B`. All public on the Hub.
- The 2026 models in Study 003: `Qwen3.5-9B-Base` (and its instruct variant)
  and `gemma-4-12B` (base, about 23 GB downloaded during that run). Loaded
  behaviourally from the Hub cache; not included here.

To obtain any of these, use the standard Hugging Face download path, for
example `AutoModelForCausalLM.from_pretrained("<model-id>")`, or
`EasyTransformer.from_pretrained("gpt2")` for the replication itself.

## Run environment and intermediate state (stripped, not science)

The following were part of the working directory but are not published, because
they are environment or process state rather than results. They regenerate
automatically and nothing in this repository depends on them being present:

- Python virtual environments (`.venv/`, `.venv-tier2/`, `.venv-2026/`) and the
  `uv`/Hugging Face/matplotlib/temp caches. Recreate from
  `run/replication/installed_packages.txt` and the per-study code.
- Generated build metadata (`easy_transformer.egg-info/`) and `__pycache__/`.
- The vendored codebase's upstream `.git/` history (the code itself is kept;
  our modifications are recorded in `run/replication/codebase.diff`).
- Engine transcripts and step logs that record the agent run rather than the
  science (`*transcript*.jsonl`, and the grading/QA side files).
