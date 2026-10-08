import os, json, time
from functools import partial
from copy import deepcopy
import torch
import numpy as np
import einops
from easy_transformer.EasyTransformer import EasyTransformer
from easy_transformer.ioi_circuit_extraction import CIRCUIT

OUT = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/ioi-gpt2-circuit/run/replication"
torch.set_grad_enabled(False)
t0 = time.time()

model = EasyTransformer.from_pretrained("gpt2").cuda()
model.set_use_attn_result(True)
model.reset_hooks()

torch.manual_seed(0)
seq_len = 100
rand_tokens = torch.randint(1000, 10000, (4, seq_len))
rand_tokens_repeat = einops.repeat(rand_tokens, "batch pos -> batch (2 pos)")

def calc_score(attn_pattern, hook, offset, arr):
    stripe = attn_pattern.diagonal(offset, dim1=-2, dim2=-1)
    scores = einops.reduce(stripe, "batch index pos -> index", "mean")
    arr[hook.layer()] = scores.detach().cpu().numpy()
    return attn_pattern

def filter_attn_hooks(hook_name):
    return hook_name.split(".")[-1] == "hook_attn"

heads_of_interest = {
    "previous": CIRCUIT["previous token"],       # [(2,2),(4,11)]
    "duplicate": CIRCUIT["duplicate token"],     # [(0,1),(0,10),(3,0)]
    "induction": CIRCUIT["induction"],           # [(5,5),(5,8),(5,9),(6,9)]
}

summary = {"seq_len": seq_len, "n_batch": 4, "arrays": {}, "scores_at_named_heads": {}, "top5_heads": {}}
for mode, offset in [("induction", 1 - seq_len), ("duplicate", -seq_len), ("previous", -1)]:
    arr = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
    model.run_with_hooks(rand_tokens_repeat,
                         fwd_hooks=[(filter_attn_hooks, partial(calc_score, offset=offset, arr=arr))])
    np.save(os.path.join(OUT, f"step6_{mode}_scores.npy"), arr)
    summary["arrays"][mode] = arr.tolist()
    # named heads for this mode's class
    named = heads_of_interest[mode]
    summary["scores_at_named_heads"][mode] = {f"{l}.{h}": float(arr[l, h]) for (l, h) in named}
    # top-5 heads overall
    flat = arr.flatten()
    idx = np.argsort(flat)[::-1][:5]
    summary["top5_heads"][mode] = [{"layer": int(i // 12), "head": int(i % 12), "score": float(flat[i])} for i in idx]
    print(f"[{mode}] named-head scores:", summary["scores_at_named_heads"][mode], flush=True)
    print(f"[{mode}] top5:", summary["top5_heads"][mode], flush=True)

summary["duration_seconds"] = time.time() - t0
with open(os.path.join(OUT, "step6_validation_result.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("DONE", flush=True)
