import os, json, time, random
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr

from easy_transformer.EasyTransformer import EasyTransformer
from easy_transformer.ioi_dataset import IOIDataset
from easy_transformer.ioi_utils import show_attention_patterns, scatter_attention_and_contribution
from easy_transformer.ioi_circuit_extraction import CIRCUIT

OUT = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/ioi-gpt2-circuit/run/replication"
torch.set_grad_enabled(False)
random.seed(0)
t0 = time.time()

model = EasyTransformer.from_pretrained("gpt2").cuda()
model.set_use_attn_result(True)
model.reset_hooks()

# ---- datasets ----
N_attn = 500
ioi_dataset = IOIDataset(prompt_type="mixed", N=N_attn, tokenizer=model.tokenizer, prepend_bos=False)

summary = {}

# (i) average attention from END to IO for the three Name Mover Heads (C5)
name_movers3 = CIRCUIT["name mover"][:3]  # [(9,9),(10,0),(9,6)]
average_attention = {}
for head in name_movers3:
    att = show_attention_patterns(model, [head], ioi_dataset, return_mtx=True, mode="attn")
    average_attention[head] = {}
    for key in ["IO", "S", "S2", "end"]:
        v = att[torch.arange(ioi_dataset.N),
                ioi_dataset.word_idx["end"],
                ioi_dataset.word_idx[key]]
        average_attention[head][key] = v.mean().item()
io_attn_per_head = {str(h): average_attention[h]["IO"] for h in name_movers3}
summary["nm_attention_on_IO_per_head"] = io_attn_per_head
summary["nm_attention_on_IO_mean_over_3heads"] = float(np.mean(list(io_attn_per_head.values())))
summary["nm_attention_full_per_head"] = {str(h): average_attention[h] for h in name_movers3}
print("(i) NM attention on IO per head:", io_attn_per_head,
      "mean:", summary["nm_attention_on_IO_mean_over_3heads"], flush=True)

# (ii) Pearson corr of attn-prob vs dot-with-name-embed for a Name Mover (9.9), N=500 (C6)
df = scatter_attention_and_contribution(model, 9, 9, ioi_dataset, return_vals=True)
x = df["Attn Prob on Name"].astype(float).values
y = df["Dot w Name Embed"].astype(float).values
rho, pval = pearsonr(x, y)
summary["scatter_head"] = "9.9"
summary["scatter_N"] = N_attn
summary["pearson_rho_attn_vs_dot"] = float(rho)
summary["pearson_pval"] = float(pval)
summary["scatter_n_points"] = int(len(x))
print(f"(ii) Pearson rho (head 9.9, N={N_attn}, {len(x)} points) = {rho:.4f} (p={pval:.2e})", flush=True)


# (iii) copy-circuit score (C7 / C8) -- reproduces experiments.py check_copy_circuit exactly
def check_copy_circuit(model, layer, head, ioi_dataset, neg=False):
    cache = {}
    model.cache_some(cache, lambda x: x == "blocks.0.hook_resid_post")
    model(ioi_dataset.toks.long())
    sign = -1 if neg else 1
    z_0 = model.blocks[1].attn.ln1(cache["blocks.0.hook_resid_post"])
    v = torch.einsum("eab,bc->eac", z_0, model.blocks[layer].attn.W_V[head])
    v += model.blocks[layer].attn.b_V[head].unsqueeze(0).unsqueeze(0)
    o = sign * torch.einsum("sph,hd->spd", v, model.blocks[layer].attn.W_O[head])
    logits = model.unembed(model.ln_final(o))
    k = 5
    n_right = 0
    for seq_idx, prompt in enumerate(ioi_dataset.ioi_prompts):
        for word in ["IO", "S", "S2"]:
            pred_tokens = [
                model.tokenizer.decode(token)
                for token in torch.topk(
                    logits[seq_idx, ioi_dataset.word_idx[word][seq_idx]], k
                ).indices
            ]
            name = "S" if "S" in word else word
            if " " + prompt[name] in pred_tokens:
                n_right += 1
    percent_right = (n_right / (ioi_dataset.N * 3)) * 100
    print(f"Copy circuit for head {layer}.{head} (sign={sign}) : Top {k} accuracy: {percent_right}%", flush=True)
    return percent_right

copy_results = {"name_movers_sign+1": {}, "negative_name_movers_sign-1": {}, "control_random_sign+1": {}}
model.reset_hooks()
print(" --- Name Mover heads (sign +1) --- ", flush=True)
for (l, h) in [(9, 9), (10, 0), (9, 6)]:
    copy_results["name_movers_sign+1"][f"{l}.{h}"] = check_copy_circuit(model, l, h, ioi_dataset, neg=False)
print(" --- Negative Name Mover heads (sign -1) --- ", flush=True)
for (l, h) in [(10, 7), (11, 10)]:
    copy_results["negative_name_movers_sign-1"][f"{l}.{h}"] = check_copy_circuit(model, l, h, ioi_dataset, neg=True)
print(" --- Control random heads (sign +1) --- ", flush=True)
for _ in range(3):
    l, h = random.randint(0, 11), random.randint(0, 11)
    copy_results["control_random_sign+1"][f"{l}.{h}"] = check_copy_circuit(model, l, h, ioi_dataset, neg=False)
summary["copy_circuit_top5_accuracy_pct"] = copy_results

summary["duration_seconds"] = time.time() - t0
with open(os.path.join(OUT, "step4_namemovers_result.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("DONE", flush=True)
