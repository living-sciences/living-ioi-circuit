"""
Tier-1 IOI circuit ladder driver (follow-up 001-living-update).

For ONE model id it computes, reusing the replication codebase verbatim where possible:
  1. C2 baseline: mean logit-diff, IO>S rate, mean IO prob (batched forward, N<=2000).
  2. C4/C9/C14/C16 discovery: the SAME generic path-patch sweep as replication step3
     (configs a,b,c,d), with the two GPT-2-small hardcodes parameterised per model.
  3. C1 discovered circuit: top-k heads per class matching the paper's class counts.
  4. C3 faithfulness: F(M), F(C), F(C)/F(M) via do_circuit_extraction keep-eval, N=100.

Writes results incrementally to results/per_model/<safe_id>.json so partial progress survives.

Usage:  python ladder_tier1.py <MODEL_ID> [configs=abcd]
"""
import os, sys, json, time
from copy import deepcopy
import torch
import numpy as np

import easy_transformer.ioi_dataset as ioi_ds_mod
from easy_transformer.EasyTransformer import EasyTransformer
from easy_transformer.ioi_dataset import IOIDataset, NAMES as ALL_NAMES
from easy_transformer.ioi_utils import logit_diff, probs, path_patching, max_2d
from easy_transformer.ioi_circuit_extraction import get_extracted_idx, do_circuit_extraction

MODEL_ID = sys.argv[1]
CONFIGS = sys.argv[2] if len(sys.argv) > 2 else "abcd"
SAFE = MODEL_ID.replace("/", "__")

OUTDIR = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/ioi-gpt2-circuit/run/followup/001-living-update/results/per_model"
os.makedirs(OUTDIR, exist_ok=True)
OUTFILE = os.path.join(OUTDIR, SAFE + ".json")

torch.set_grad_enabled(False)
t0 = time.time()

# paper / GPT-2-small class counts (the target sizes for the discovered circuit)
PAPER_COUNTS = {"name mover": 11, "negative": 2, "s2 inhibition": 4,
                "induction": 4, "duplicate token": 3, "previous token": 2}
CLASS_TOKENS = {"name mover": ["end"], "negative": ["end"], "s2 inhibition": ["end"],
                "induction": ["S2"], "duplicate token": ["S2"], "previous token": ["S+1"]}

res = {"model_id": MODEL_ID, "configs_requested": CONFIGS}

# ---- Pythia registration (once, before load) ----
EasyTransformer.VALID_PRETRAINED_MODEL_NAMES |= {
    "EleutherAI/pythia-70m", "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-160m", "EleutherAI/pythia-1b"}

print(f"[{MODEL_ID}] loading ...", flush=True)
model = EasyTransformer.from_pretrained(MODEL_ID).cuda()
model.set_use_attn_result(True)
model.set_use_headwise_qkv_input(False)  # only turned on for configs b/c/d (qkv receivers); off for a to save memory
n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
res["n_layers"], res["n_heads"] = int(n_layers), int(n_heads)
res["positional_embedding_type"] = model.cfg.positional_embedding_type
print(f"[{MODEL_ID}] loaded n_layers={n_layers} n_heads={n_heads} pe={model.cfg.positional_embedding_type}", flush=True)

# ---- single-token-name filter (mandatory for non-GPT-2 tokenizers) ----
names1 = [n for n in ALL_NAMES if len(model.tokenizer(" " + n).input_ids) == 1]
res["n_names_total"] = len(ALL_NAMES)
res["n_names_single_token"] = len(names1)
print(f"[{MODEL_ID}] single-token names: {len(names1)}/{len(ALL_NAMES)}", flush=True)
# monkeypatch the module-global NAMES used by gen_prompt_uniform + gen_flipped_prompts
ioi_ds_mod.NAMES = names1

if len(names1) < 15:
    res["status"] = "not attempted (<15 single-token names)"
    res["duration_seconds"] = time.time() - t0
    json.dump(res, open(OUTFILE, "w"), indent=2)
    print(f"[{MODEL_ID}] <15 names -> not attempted", flush=True)
    sys.exit(0)

def save():
    res["duration_seconds"] = time.time() - t0
    json.dump(res, open(OUTFILE, "w"), indent=2)

# ================= 1. BASELINE (C2) =================
N_base = 2000
d = IOIDataset(prompt_type="mixed", N=N_base, tokenizer=model.tokenizer, prepend_bos=False)
toks = d.toks.long().cuda()
end_idx = torch.as_tensor(d.word_idx["end"]).long()
io_ids = torch.as_tensor(d.io_tokenIDs).long()
s_ids = torch.as_tensor(d.s_tokenIDs).long()
bs = 200
ld_all = torch.empty(N_base); ioprob_all = torch.empty(N_base)
for i in range(0, N_base, bs):
    j = min(i + bs, N_base)
    logits = model(toks[i:j]).detach()
    ar = torch.arange(j - i)
    e = end_idx[i:j]
    el = logits[ar, e, :]
    ld_all[i:j] = (el[ar, io_ids[i:j]] - el[ar, s_ids[i:j]]).cpu()
    ioprob_all[i:j] = torch.softmax(el, dim=1)[ar, io_ids[i:j]].cpu()
res["baseline"] = {
    "N": N_base,
    "mean_logit_diff": float(ld_all.mean()),
    "std_logit_diff": float(ld_all.std()),
    "io_over_s_rate": float((ld_all > 0).float().mean()),
    "mean_io_prob": float(ioprob_all.mean()),
}
print(f"[{MODEL_ID}] BASELINE ld={res['baseline']['mean_logit_diff']:.4f} "
      f"IO>S={res['baseline']['io_over_s_rate']:.4f} ioprob={res['baseline']['mean_io_prob']:.4f}", flush=True)
save()

torch.cuda.empty_cache()

# ================= 2. DISCOVERY (path-patch sweeps) =================
# adaptive N: the qkv-input caching in configs b/c/d scales with n_layers*n_heads*d_model
mem_proxy = n_layers * n_heads * model.cfg.d_model
N_pp = 100 if mem_proxy < 400_000 else (50 if mem_proxy < 900_000 else 30)
res["N_pp"] = N_pp
print(f"[{MODEL_ID}] mem_proxy={mem_proxy} -> N_pp={N_pp}", flush=True)
ioi_dataset = IOIDataset(prompt_type="mixed", N=N_pp, tokenizer=model.tokenizer, prepend_bos=False)
abc_dataset = (ioi_dataset.gen_flipped_prompts(("IO", "RAND"))
               .gen_flipped_prompts(("S", "RAND"))
               .gen_flipped_prompts(("S1", "RAND")))

def run_sweep(receiver_hooks, position, cache_npy=None):
    model.reset_hooks()
    torch.cuda.empty_cache()
    default_ld = logit_diff(model, ioi_dataset)
    if cache_npy is not None and os.path.exists(cache_npy):
        print(f"[{MODEL_ID}] reuse cached matrix {os.path.basename(cache_npy)}", flush=True)
        return torch.from_numpy(np.load(cache_npy)), float(default_ld)
    results = torch.zeros(size=(n_layers, n_heads))
    for sl in range(n_layers):
        for sh in range(n_heads):
            model.reset_hooks()
            path_patching(model=model, D_new=abc_dataset, D_orig=ioi_dataset,
                          sender_heads=[(sl, sh)], receiver_hooks=receiver_hooks,
                          positions=[position], return_hooks=False,
                          freeze_mlps=False, have_internal_interactions=False)
            results[sl][sh] = logit_diff(model, ioi_dataset) - default_ld
    model.reset_hooks()
    results = 100.0 * results / default_ld
    return results, float(default_ld)

def topk_signed(mat, k, most_negative=True):
    """top-k heads by signed effect.
    most_negative=True  -> most negative pct_change first (true contributors: name movers,
                           s-inhibition, induction, duplicate, previous token).
    most_negative=False -> most positive first (negative name movers)."""
    flat = [((l, h), float(mat[l, h])) for l in range(mat.shape[0]) for h in range(mat.shape[1])]
    flat.sort(key=lambda x: x[1], reverse=(not most_negative))
    return [{"layer": l, "head": h, "pct_change": v} for (l, h), v in flat[:k]]

disc = {"N": N_pp}
circuit = {}  # class -> list[[l,h]]

# config a: h -> Logits @ END (receiver = last block resid_post)
if "a" in CONFIGS:
    ta = time.time()
    cache_a = os.path.join(OUTDIR, f"{SAFE}_a_heads.npy")
    mat_a, dl_a = run_sweep([(f"blocks.{n_layers-1}.hook_resid_post", None)], "end", cache_npy=cache_a)
    np.save(cache_a, mat_a.numpy())
    nm = topk_signed(mat_a, PAPER_COUNTS["name mover"], most_negative=True)   # name movers = most negative
    negnm = topk_signed(mat_a, PAPER_COUNTS["negative"], most_negative=False) # negative NM = most positive
    disc["a_h_to_logits"] = {"default_logit_diff": dl_a, "seconds": time.time()-ta,
                             "name_movers_top": nm, "negNM_top": negnm}
    circuit["name mover"] = [[h["layer"], h["head"]] for h in nm]
    circuit["negative"] = [[h["layer"], h["head"]] for h in negnm]
    print(f"[{MODEL_ID}] config a done {time.time()-ta:.1f}s NM={circuit['name mover'][:3]} negNM={circuit['negative']}", flush=True)
    res["discovery"] = disc; save()

# configs b/c/d use per-head q/k/v receivers -> need headwise qkv-input caching enabled
if any(c in CONFIGS for c in "bcd"):
    model.set_use_headwise_qkv_input(True)
    torch.cuda.empty_cache()

# config b: h -> discovered-NM queries @ END  -> S-inhibition
if "b" in CONFIGS and circuit.get("name mover"):
    tb = time.time()
    nm_heads = circuit["name mover"][:3] if len(circuit["name mover"]) >= 3 else circuit["name mover"]
    recv_b = [(f"blocks.{l}.attn.hook_q", h) for l, h in nm_heads]
    cache_b = os.path.join(OUTDIR, f"{SAFE}_b_heads.npy")
    mat_b, dl_b = run_sweep(recv_b, "end", cache_npy=cache_b)
    np.save(cache_b, mat_b.numpy())
    sinh = topk_signed(mat_b, PAPER_COUNTS["s2 inhibition"], most_negative=True)
    disc["b_h_to_NMquery"] = {"default_logit_diff": dl_b, "seconds": time.time()-tb,
                              "nm_query_receivers": nm_heads, "s_inhibition_top": sinh}
    circuit["s2 inhibition"] = [[h["layer"], h["head"]] for h in sinh]
    print(f"[{MODEL_ID}] config b done {time.time()-tb:.1f}s Sinh={circuit['s2 inhibition']}", flush=True)
    res["discovery"] = disc; save()

# config c: h -> discovered-S-inh values @ S2 -> induction + duplicate
if "c" in CONFIGS and circuit.get("s2 inhibition"):
    tc = time.time()
    recv_c = [(f"blocks.{l}.attn.hook_v", h) for l, h in circuit["s2 inhibition"]]
    cache_c = os.path.join(OUTDIR, f"{SAFE}_c_heads.npy")
    mat_c, dl_c = run_sweep(recv_c, "S2", cache_npy=cache_c)
    np.save(cache_c, mat_c.numpy())
    n_valsend = PAPER_COUNTS["induction"] + PAPER_COUNTS["duplicate token"]  # 7
    val_senders = topk_signed(mat_c, n_valsend, most_negative=True)
    # heuristic split: earliest-layer heads -> duplicate token; rest -> induction
    by_layer = sorted(val_senders, key=lambda x: x["layer"])
    dup = by_layer[:PAPER_COUNTS["duplicate token"]]
    ind = by_layer[PAPER_COUNTS["duplicate token"]:]
    disc["c_h_to_SInhval"] = {"default_logit_diff": dl_c, "seconds": time.time()-tc,
                              "value_senders_top": val_senders,
                              "split_note": "earliest-layer -> duplicate token, deeper -> induction (heuristic)"}
    circuit["duplicate token"] = [[h["layer"], h["head"]] for h in dup]
    circuit["induction"] = [[h["layer"], h["head"]] for h in ind]
    print(f"[{MODEL_ID}] config c done {time.time()-tc:.1f}s ind={circuit['induction']} dup={circuit['duplicate token']}", flush=True)
    res["discovery"] = disc; save()

# config d: h -> discovered-induction keys @ S+1 -> previous token
if "d" in CONFIGS and circuit.get("induction"):
    td = time.time()
    recv_d = [(f"blocks.{l}.attn.hook_k", h) for l, h in circuit["induction"]]
    cache_d = os.path.join(OUTDIR, f"{SAFE}_d_heads.npy")
    mat_d, dl_d = run_sweep(recv_d, "S+1", cache_npy=cache_d)
    np.save(cache_d, mat_d.numpy())
    prev = topk_signed(mat_d, PAPER_COUNTS["previous token"], most_negative=True)
    disc["d_h_to_INDkey"] = {"default_logit_diff": dl_d, "seconds": time.time()-td,
                             "previous_token_top": prev}
    circuit["previous token"] = [[h["layer"], h["head"]] for h in prev]
    print(f"[{MODEL_ID}] config d done {time.time()-td:.1f}s prev={circuit['previous token']}", flush=True)
    res["discovery"] = disc; save()

res["discovered_circuit"] = circuit
res["discovered_circuit_counts"] = {k: len(v) for k, v in circuit.items()}
res["discovered_circuit_total_heads"] = sum(len(v) for v in circuit.values())
res["discovered_circuit_num_classes"] = len(circuit)
save()

# ================= 3. FAITHFULNESS (C3) =================
# build heads_to_keep directly from the discovered circuit (bypass global RELEVANT_TOKENS)
model.set_use_headwise_qkv_input(False)  # faithfulness uses head "result" only; free qkv-input caches
model.reset_hooks()
torch.cuda.empty_cache()
F_M = logit_diff(model, ioi_dataset).item()
heads_to_keep = {}
for cls, heads in circuit.items():
    for h in heads:
        heads_to_keep[tuple(h)] = get_extracted_idx(CLASS_TOKENS[cls], ioi_dataset)
model.reset_hooks()
model2, _ = do_circuit_extraction(model=model, heads_to_keep=heads_to_keep,
                                  mlps_to_remove={}, ioi_dataset=ioi_dataset,
                                  mean_dataset=abc_dataset)
F_C = logit_diff(model2, ioi_dataset).item()
model.reset_hooks()
res["faithfulness"] = {
    "N": N_pp, "F_M": F_M, "F_C": F_C,
    "abs_F_M_minus_F_C": abs(F_M - F_C),
    "F_C_over_F_M": (F_C / F_M) if F_M != 0 else None,
    "n_heads_in_circuit": len(heads_to_keep),
    "classes_included": sorted(set(cls for cls in circuit if circuit[cls])),
}
print(f"[{MODEL_ID}] FAITHFULNESS F(M)={F_M:.4f} F(C)={F_C:.4f} F(C)/F(M)={F_C/F_M:.4f} "
      f"({len(heads_to_keep)} heads)", flush=True)
res["status"] = "ok"
save()
print(f"[{MODEL_ID}] DONE in {time.time()-t0:.1f}s -> {OUTFILE}", flush=True)
