import os, json, time
from copy import deepcopy
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from easy_transformer.EasyTransformer import EasyTransformer
from easy_transformer.ioi_dataset import IOIDataset
from easy_transformer.ioi_utils import logit_diff
from easy_transformer.ioi_circuit_extraction import (
    CIRCUIT, get_heads_circuit, do_circuit_extraction,
)

OUT = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/ioi-gpt2-circuit/run/replication"
SVG = os.path.join(OUT, "svgs"); os.makedirs(SVG, exist_ok=True)
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
torch.set_grad_enabled(False)
t0 = time.time()

model = EasyTransformer.from_pretrained("gpt2").cuda()
model.set_use_attn_result(True)

N = 100
ioi_dataset = IOIDataset(prompt_type="mixed", N=N, tokenizer=model.tokenizer)
abc_dataset = (
    ioi_dataset.gen_flipped_prompts(("IO", "RAND"))
    .gen_flipped_prompts(("S", "RAND"))
    .gen_flipped_prompts(("S1", "RAND"))
)
mean_dataset = abc_dataset
circuit = deepcopy(CIRCUIT)
metric = logit_diff

# ---- Build K sets (verbatim from minimality.py) ----
K = {}
for circuit_class in circuit.keys():
    for head in circuit[circuit_class]:
        K[head] = [circuit_class]
for head in K.keys():
    new_j_entry = []
    for entry in K[head]:
        if isinstance(entry, str):
            for head2 in CIRCUIT[entry]:
                new_j_entry.append(head2)
        elif isinstance(entry, tuple):
            new_j_entry.append(entry)
    assert head in new_j_entry, (head, new_j_entry)
    K[head] = list(set(new_j_entry))
# name mover ordering special-case
for i, head in enumerate(circuit["name mover"]):
    K[head] = deepcopy(circuit["name mover"][: i + 1])
for head in [(9, 0), (11, 9)]:
    K[head] = circuit["name mover"] + circuit["negative"]
K[(5, 8)] = [(11, 10), (10, 7), (5, 8)]
K[(5, 9)] = [(11, 10), (10, 7), (5, 9)]

# ---- Run experiment ----
results = {}
results_cache = {}
for circuit_class in circuit.keys():
    for head in circuit[circuit_class]:
        results[head] = [None, None]
        base = frozenset(K[head])
        summit_list = deepcopy(K[head]); summit_list.remove(head)
        summit = frozenset(summit_list)
        for idx, ablated_stuff in enumerate([base, summit]):
            if ablated_stuff not in results_cache:
                new_heads_to_keep = get_heads_circuit(ioi_dataset, excluded=ablated_stuff, circuit=circuit)
                model.reset_hooks()
                model, _ = do_circuit_extraction(model=model, heads_to_keep=new_heads_to_keep,
                                                 mlps_to_remove={}, ioi_dataset=ioi_dataset, mean_dataset=mean_dataset)
                results_cache[ablated_stuff] = float(metric(model, ioi_dataset, std=False))
            results[head][idx] = results_cache[ablated_stuff]
        print(f"{head} K-size {len(K[head])}: F(C\\K)={results[head][0]:.3f} -> F(C\\(K\\v))={results[head][1]:.3f} "
              f"minimality={abs(results[head][1]-results[head][0]):.3f}", flush=True)

model.reset_hooks()

# ---- Assemble bar chart data (name mover split into main + backup) ----
relevant_classes = list(circuit.keys())
out = {"N": N, "results": {}, "minimality_scores": {}, "by_class": {}}
labels, heights, colors = [], [], []
palette = {"name mover": "#636EFA", "negative": "#EF553B", "s2 inhibition": "#00CC96",
           "induction": "#AB63FA", "duplicate token": "#FFA15A", "previous token": "#19D3F3",
           "backup name mover": "#FF6692"}
for head in results:
    out["results"][str(head)] = results[head]
    out["minimality_scores"][str(head)] = abs(results[head][1] - results[head][0])

for G in relevant_classes + ["backup name mover"]:
    if G == "backup name mover":
        curvys = [h for h in circuit["name mover"] if h not in [(9, 6), (9, 9), (10, 0)]]
    elif G == "name mover":
        curvys = [(9, 6), (9, 9), (10, 0)]
    else:
        curvys = list(circuit[G])
    curvys = sorted(curvys, key=lambda x: -abs(results[x][1] - results[x][0]))
    out["by_class"][G] = [{"head": str(v), "minimality": abs(results[v][1] - results[v][0])} for v in curvys]
    for v in curvys:
        labels.append(f"{v[0]}.{v[1]}")
        heights.append(abs(results[v][1] - results[v][0]))
        colors.append(palette[G])

fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(range(len(labels)), heights, color=colors)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90)
ax.set_xlabel("Attention head"); ax.set_ylabel("|change in logit difference| (minimality score)")
ax.set_title("Circuit minimality: marginal effect of each head")
fig.tight_layout()
fig.savefig(os.path.join(SVG, "circuit_minimality.svg"))
fig.savefig(os.path.join(FIG, "step9_minimality.png"), dpi=120)
plt.close(fig)

scores = list(out["minimality_scores"].values())
out["min_score"] = float(min(scores))
out["max_score"] = float(max(scores))
out["all_nonzero"] = bool(all(s > 1e-6 for s in scores))
out["duration_seconds"] = time.time() - t0
with open(os.path.join(OUT, "step9_minimality_result.json"), "w") as f:
    json.dump(out, f, indent=2)
print(f"DONE: {len(scores)} heads, min score={out['min_score']:.4f}, max={out['max_score']:.4f}, all nonzero={out['all_nonzero']}", flush=True)
