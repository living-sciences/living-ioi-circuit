import os, json, time
import torch
import numpy as np
from tqdm import tqdm
from easy_transformer.EasyTransformer import EasyTransformer
from easy_transformer.ioi_dataset import IOIDataset
from easy_transformer.ioi_utils import logit_diff, path_patching, max_2d
from easy_transformer.ioi_circuit_extraction import do_circuit_extraction, get_heads_circuit, CIRCUIT

OUT = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/ioi-gpt2-circuit/run/replication"
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
torch.set_grad_enabled(False)
t0 = time.time()

model = EasyTransformer.from_pretrained("gpt2").cuda()
model.set_use_headwise_qkv_input(True)
model.set_use_attn_result(True)

N = 100
ioi_dataset = IOIDataset(prompt_type="mixed", N=N, tokenizer=model.tokenizer, prepend_bos=False)
abc_dataset = (
    ioi_dataset.gen_flipped_prompts(("IO", "RAND"))
    .gen_flipped_prompts(("S", "RAND"))
    .gen_flipped_prompts(("S1", "RAND"))
)

model.reset_hooks()
default_logit_diff = logit_diff(model, ioi_dataset)
print("baseline logit diff:", default_logit_diff.item(), flush=True)

top_name_movers = [(9, 9), (9, 6), (10, 0)]
exclude_heads = [(l, h) for l in range(12) for h in range(12)]
for head in top_name_movers:
    exclude_heads.remove(head)

the_extra_hooks = do_circuit_extraction(
    model=model,
    heads_to_remove=get_heads_circuit(ioi_dataset=ioi_dataset, circuit={"name mover": top_name_movers}),
    mlps_to_remove={},
    ioi_dataset=ioi_dataset,
    mean_dataset=abc_dataset,
    return_hooks=True,
    excluded=exclude_heads,
)
model.reset_hooks()
for hook in the_extra_hooks:
    model.add_hook(*hook)
knockout_logit_diff = logit_diff(model, ioi_dataset)
model.reset_hooks()
print("post-NM-knockout logit diff:", knockout_logit_diff.item(), flush=True)

both_results = {}
pos = "end"
for idx, extra_hooks in enumerate([[], the_extra_hooks]):
    results = torch.zeros(size=(12, 12))
    model.reset_hooks()
    for hook in extra_hooks:
        model.add_hook(*hook)
    hooked_ld = logit_diff(model, ioi_dataset)
    model.reset_hooks()
    for source_layer in tqdm(range(12)):
        for source_head_idx in range(12):
            model.reset_hooks()
            model = path_patching(
                model=model, D_new=abc_dataset, D_orig=ioi_dataset,
                sender_heads=[(source_layer, source_head_idx)],
                receiver_hooks=[("blocks.11.hook_resid_post", None)],
                positions=[pos], return_hooks=False, extra_hooks=extra_hooks,
            )
            cur = logit_diff(model, ioi_dataset)
            results[source_layer][source_head_idx] = cur - hooked_ld
    model.reset_hooks()
    key = "baseline" if idx == 0 else "nm_knocked_out"
    both_results[key] = results.clone()
    np.save(os.path.join(OUT, f"step7_direct_effect_{key}.npy"), results.numpy())

# top heads in NM-knocked-out sweep (by |effect|)
res_ko = both_results["nm_knocked_out"]
coords, vals = max_2d(res_ko.abs(), k=10)
top_ko = [{"layer": int(l), "head": int(h), "effect": float(res_ko[l, h])} for (l, h) in coords]

# reference: baseline top
res_b = both_results["baseline"]
coords_b, _ = max_2d(res_b.abs(), k=10)
top_b = [{"layer": int(l), "head": int(h), "effect": float(res_b[l, h])} for (l, h) in coords_b]

summary = {
    "N": N,
    "baseline_logit_diff": float(default_logit_diff),
    "post_NM_knockout_logit_diff": float(knockout_logit_diff),
    "relative_drop_pct": float(100 * (default_logit_diff - knockout_logit_diff) / default_logit_diff),
    "top10_direct_effect_baseline": top_b,
    "top10_direct_effect_NM_knocked_out": top_ko,
    "duration_seconds": time.time() - t0,
}
with open(os.path.join(OUT, "step7_backupnm_result.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("baseline top10:", top_b, flush=True)
print("NM-knocked-out top10 (Backup Name Movers emerge):", top_ko, flush=True)
print("DONE", flush=True)
