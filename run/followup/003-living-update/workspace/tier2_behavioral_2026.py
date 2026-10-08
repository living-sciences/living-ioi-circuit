"""Tier-2 behavioral-only IOI baseline for 2026 MULTIMODAL-arch models (Qwen3.5-9B, gemma-4-12B).

These flagships cannot be wrapped by the easy_transformer fork / TransformerLens, and even
nnsight cannot wrap them; per the 003 instruction we load them with plain transformers via
AutoModelForImageTextToText and score the SAME IOI test used for the Tier-2 modern models in 001.

Reuses the SAME IOI prompt strings (data/ioi_prompts.json) and the SAME single-token-name filter
+ END-scoring logic as 001/tier2_behavioral.py. Only the loader class differs (multimodal arch:
AutoModelForImageTextToText; the text logits still come straight off m(**ids).logits).

Usage: python tier2_behavioral_2026.py <MODEL_PATH> <FRIENDLY_ID>
  e.g. python tier2_behavioral_2026.py /.../Qwen3.5-9B-Base Qwen/Qwen3.5-9B-Base
"""
import os, sys, json, time
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL_PATH = sys.argv[1]
MODEL_ID = sys.argv[2]                      # friendly id used as the ladder label
SAFE = MODEL_ID.replace("/", "__")
OUTDIR = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/ioi-gpt2-circuit/run/followup/003-living-update/results/per_model"
os.makedirs(OUTDIR, exist_ok=True)
OUTFILE = os.path.join(OUTDIR, f"tier2_{SAFE}.json")

DATA = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/ioi-gpt2-circuit/run/followup/003-living-update/workspace/data/ioi_prompts.json"
prompts = json.load(open(DATA))
t0 = time.time()
torch.set_grad_enabled(False)

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
# 2026 multimodal arch: load the whole conditional-generation model; text logits come off m(**ids).logits
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map="auto").eval()
dev = next(model.parameters()).device

# single-token name filter in THIS model's tokenizer (MANDATORY, same rule as 001)
def one_tok(name):
    ids = tok(" " + name, add_special_tokens=False).input_ids
    return len(ids) == 1
all_names = sorted(set([p["IO"] for p in prompts] + [p["S"] for p in prompts]))
names1 = set(n for n in all_names if one_tok(n))
res = {"model_id": MODEL_ID, "model_path": MODEL_PATH, "tier": 2, "arch": type(model).__name__,
       "loader": "AutoModelForImageTextToText", "n_names_in_prompts": len(all_names),
       "n_names_single_token": len(names1)}
print(f"[{MODEL_ID}] single-token names among prompt names: {len(names1)}/{len(all_names)}", flush=True)

if len(names1) < 15:
    res["status"] = "not attempted (<15 single-token names)"
    res["duration_seconds"] = time.time() - t0
    json.dump(res, open(OUTFILE, "w"), indent=2)
    print(f"[{MODEL_ID}] <15 names -> not attempted", flush=True); sys.exit(0)

lds, io_probs, io_gt_s = [], [], []
used = 0
add_bos = tok.bos_token_id is not None
for p in prompts:
    IO, S = p["IO"], p["S"]
    if IO not in names1 or S not in names1:
        continue
    assert p["text"].endswith(" " + IO)
    query = p["text"][: -(len(" " + IO))]   # sentence up to the final " to"
    ids = tok(query, add_special_tokens=add_bos, return_tensors="pt").input_ids.to(dev)
    logits = model(input_ids=ids).logits[0, -1, :].float()   # next-token dist at END
    io_id = tok(" " + IO, add_special_tokens=False).input_ids[0]
    s_id = tok(" " + S, add_special_tokens=False).input_ids[0]
    lds.append(float(logits[io_id] - logits[s_id]))
    probs = torch.softmax(logits, dim=-1)
    io_probs.append(float(probs[io_id]))
    io_gt_s.append(1.0 if logits[io_id] > logits[s_id] else 0.0)
    used += 1

import statistics as st
res["baseline"] = {
    "N_scored": used,
    "mean_logit_diff": st.fmean(lds),
    "std_logit_diff": st.pstdev(lds),
    "io_over_s_rate": st.fmean(io_gt_s),
    "mean_io_prob": st.fmean(io_probs),
}
res["status"] = "ok"
res["duration_seconds"] = time.time() - t0
json.dump(res, open(OUTFILE, "w"), indent=2)
print(f"[{MODEL_ID}] BEHAVIORAL N={used} ld={res['baseline']['mean_logit_diff']:+.4f} "
      f"IO>S={res['baseline']['io_over_s_rate']:.4f} ioprob={res['baseline']['mean_io_prob']:.4f} "
      f"({res['duration_seconds']:.1f}s)", flush=True)
