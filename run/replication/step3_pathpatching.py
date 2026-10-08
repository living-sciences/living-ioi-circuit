import os, json, time
from copy import deepcopy
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from easy_transformer.EasyTransformer import EasyTransformer
from easy_transformer.ioi_dataset import IOIDataset
from easy_transformer.ioi_utils import logit_diff, path_patching, max_2d
from easy_transformer.ioi_circuit_extraction import CIRCUIT

OUT = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/ioi-gpt2-circuit/run/replication"
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
torch.set_grad_enabled(False)
t0 = time.time()

model = EasyTransformer.from_pretrained("gpt2").cuda()
model.set_use_headwise_qkv_input(True)
model.set_use_attn_result(True)

N = 100  # shipped default in experiments.py for path patching sweeps
ioi_dataset = IOIDataset(prompt_type="mixed", N=N, tokenizer=model.tokenizer, prepend_bos=False)
abc_dataset = (
    ioi_dataset.gen_flipped_prompts(("IO", "RAND"))
    .gen_flipped_prompts(("S", "RAND"))
    .gen_flipped_prompts(("S1", "RAND"))
)
circuit = deepcopy(CIRCUIT)
print("Setup done in %.1fs, N=%d" % (time.time()-t0, N), flush=True)


def run_sweep(receiver_hooks, position, title, tag):
    model.reset_hooks()
    default_logit_diff = logit_diff(model, ioi_dataset)
    results = torch.zeros(size=(12, 12))
    mlp_results = torch.zeros(size=(12, 1))
    for source_layer in tqdm(range(12)):
        for source_head_idx in [None] + list(range(12)):
            model.reset_hooks()
            path_patching(
                model=model,
                D_new=abc_dataset,
                D_orig=ioi_dataset,
                sender_heads=[(source_layer, source_head_idx)],
                receiver_hooks=receiver_hooks,
                positions=[position],
                return_hooks=False,
                freeze_mlps=False,
                have_internal_interactions=False,
            )
            cur_logit_diff = logit_diff(model, ioi_dataset)
            if source_head_idx is None:
                mlp_results[source_layer] = cur_logit_diff - default_logit_diff
            else:
                results[source_layer][source_head_idx] = cur_logit_diff - default_logit_diff
    model.reset_hooks()
    results = 100.0 * results / default_logit_diff
    mlp_results = 100.0 * mlp_results / default_logit_diff

    # save tensors
    np.save(os.path.join(OUT, f"step3_{tag}_heads.npy"), results.numpy())
    np.save(os.path.join(OUT, f"step3_{tag}_mlp.npy"), mlp_results.numpy())

    # heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(results.numpy(), cmap="RdBu", vmin=-abs(results).max().item(),
                   vmax=abs(results).max().item(), origin="upper")
    ax.set_xlabel("head"); ax.set_ylabel("layer"); ax.set_title(title)
    fig.colorbar(im, ax=ax, label="% change in logit difference")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"step3_{tag}.png"), dpi=120); plt.close(fig)

    # top heads by |effect|
    coords, vals = max_2d(results.abs(), k=8)
    tops = []
    for (l, h) in coords:
        tops.append({"layer": int(l), "head": int(h),
                     "pct_change": float(results[l, h])})
    print(f"[{tag}] default_logit_diff={default_logit_diff.item():.4f} top8={tops}", flush=True)
    return {"default_logit_diff": float(default_logit_diff),
            "top8_heads_by_abs_effect": tops}


name_movers = [(9, 9), (9, 6), (10, 0)]
induction = circuit["induction"]  # [(5,5),(5,8),(5,9),(6,9)]

configs = {
    "a_h_to_logits": dict(
        receiver_hooks=[("blocks.11.hook_resid_post", None)], position="end",
        title="(a) Heads -> Logits (Name Movers / Neg NM), END"),
    "b_h_to_NMquery": dict(
        receiver_hooks=[(f"blocks.{l}.attn.hook_q", h) for l, h in name_movers], position="end",
        title="(b) Heads -> Name Mover queries (S-Inhibition), END"),
    "c_h_to_SInhval": dict(
        receiver_hooks=[(f"blocks.{l}.attn.hook_v", h) for l, h in circuit["s2 inhibition"]], position="S2",
        title="(c) Heads -> S-Inhibition values (Dup Tok/Induction), S2"),
    "d_h_to_INDkey": dict(
        receiver_hooks=[(f"blocks.{l}.attn.hook_k", h) for l, h in induction], position="S+1",
        title="(d) Heads -> Induction keys (Previous Token), S1+1"),
}

summary = {"N": N, "configs": {}}
for tag, cfg in configs.items():
    print("=== running", tag, "===", flush=True)
    summary["configs"][tag] = run_sweep(cfg["receiver_hooks"], cfg["position"], cfg["title"], tag)

summary["CIRCUIT"] = {k: [list(t) for t in v] for k, v in CIRCUIT.items()}
summary["CIRCUIT_counts"] = {k: len(v) for k, v in CIRCUIT.items()}
summary["CIRCUIT_total_heads"] = sum(len(v) for v in CIRCUIT.values())
summary["CIRCUIT_num_categories"] = len(CIRCUIT)
summary["duration_seconds"] = time.time() - t0
with open(os.path.join(OUT, "step3_pathpatching_result.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("DONE", json.dumps(summary["CIRCUIT_counts"]), "total", summary["CIRCUIT_total_heads"], flush=True)
