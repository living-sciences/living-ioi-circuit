import os, json, time
import torch
import numpy as np
from easy_transformer.EasyTransformer import EasyTransformer
from easy_transformer.ioi_dataset import IOIDataset
from easy_transformer.ioi_utils import show_attention_patterns, path_patching, logit_diff
from easy_transformer.ioi_circuit_extraction import CIRCUIT
from easy_transformer.experiments import ExperimentMetric, PatchingConfig, EasyPatching

OUT = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/ioi-gpt2-circuit/run/replication"
torch.set_grad_enabled(False)
t0 = time.time()

model = EasyTransformer.from_pretrained("gpt2").cuda()
model.set_use_headwise_qkv_input(True)
model.set_use_attn_result(True)

N = 500
ioi_dataset = IOIDataset(prompt_type="mixed", N=N, tokenizer=model.tokenizer, prepend_bos=False)
abc_dataset = (
    ioi_dataset.gen_flipped_prompts(("IO", "RAND"))
    .gen_flipped_prompts(("S", "RAND"))
    .gen_flipped_prompts(("S1", "RAND"))
)
name_movers3 = CIRCUIT["name mover"][:3]           # [(9,9),(10,0),(9,6)]
s_inhibition_heads = [(8, 6), (8, 10), (7, 3), (7, 9)]
summary = {"N": N}


def nm_attention(model, keys=("IO", "S", "S2")):
    """avg attention END->key averaged over the 3 name movers, with current hooks active"""
    out = {k: [] for k in keys}
    for head in name_movers3:
        att = show_attention_patterns(model, [head], ioi_dataset, return_mtx=True, mode="attn")
        for k in keys:
            v = att[torch.arange(ioi_dataset.N), ioi_dataset.word_idx["end"], ioi_dataset.word_idx[k]]
            out[k].append(v.mean().item())
    return {k: float(np.mean(out[k])) for k in keys}, {str(name_movers3[i]): {k: out[k][i] for k in keys} for i in range(len(name_movers3))}


# (i) NM attention before vs after path-patching S-Inhibition heads -> NM queries (from pABC)
model.reset_hooks()
before_mean, before_per = nm_attention(model)
model.reset_hooks()
model = path_patching(
    model=model, D_new=abc_dataset, D_orig=ioi_dataset,
    sender_heads=s_inhibition_heads,
    receiver_hooks=[(f"blocks.{l}.attn.hook_q", h) for (l, h) in name_movers3],
    positions=["end"], return_hooks=False,
)
after_mean, after_per = nm_attention(model)
model.reset_hooks()
summary["nm_attention_before_patch"] = before_mean
summary["nm_attention_after_patch"] = after_mean
summary["nm_attention_before_per_head"] = before_per
summary["nm_attention_after_per_head"] = after_per
print("(i) NM attention BEFORE:", before_mean, flush=True)
print("(i) NM attention AFTER  :", after_mean, flush=True)

# (ii) avg END->S2 attention over the 4 S-Inhibition heads (clean model)
model.reset_hooks()
s2_atts = {}
for head in s_inhibition_heads:
    att = show_attention_patterns(model, [head], ioi_dataset, return_mtx=True, mode="attn")
    v = att[torch.arange(ioi_dataset.N), ioi_dataset.word_idx["end"], ioi_dataset.word_idx["S2"]]
    s2_atts[str(head)] = v.mean().item()
summary["s_inhibition_END_to_S2_per_head"] = s2_atts
summary["s_inhibition_END_to_S2_mean"] = float(np.mean(list(s2_atts.values())))
print("(ii) S-Inhibition END->S2 per head:", s2_atts, "mean:", summary["s_inhibition_END_to_S2_mean"], flush=True)

# (iii) Token and position signal results -> 3x2 logit_diff_per_signal
ssd = {}
ssd[(0, 1)] = ioi_dataset.gen_flipped_prompts(("IO", "RAND")).gen_flipped_prompts(("S", "RAND"))
ssd[(0, -1)] = ssd[(0, 1)].gen_flipped_prompts(("IO", "S1"))
ssd[(-1, -1)] = ioi_dataset.gen_flipped_prompts(("S2", "IO"))
ssd[(-1, 1)] = ssd[(-1, -1)].gen_flipped_prompts(("IO", "S1"))
ssd[(1, -1)] = ioi_dataset.gen_flipped_prompts(("IO", "S1"))
ssd[(1, 1)] = ioi_dataset

def patch_end(z, source_act, hook):
    z[torch.arange(ioi_dataset.N), ioi_dataset.word_idx["end"]] = source_act[
        torch.arange(ioi_dataset.N), ioi_dataset.word_idx["end"]]
    return z

logit_diff_per_signal = np.zeros((3, 2))
for k, source_dataset in ssd.items():
    config = PatchingConfig(
        source_dataset=source_dataset.toks.long(),
        target_dataset=ioi_dataset.toks.long(),
        target_module="attn_head", head_circuit="result",
        cache_act=True, verbose=False, patch_fn=patch_end, layers=(0, 9 - 1),
    )
    metric = ExperimentMetric(lambda x: x, ioi_dataset)
    patching = EasyPatching(model, config, metric)
    model.reset_hooks()
    for l, h in s_inhibition_heads:
        hk_name, hk = patching.get_hook(l, h)
        model.add_hook(hk_name, hk)
    tok_s, pos_s = k
    logit_diff_per_signal[tok_s + 1, (pos_s + 1) // 2] = logit_diff(model, ioi_dataset)
model.reset_hooks()

summary["logit_diff_per_signal_rows_tokensignal"] = ["inverted(-1)", "uncorrelated(0)", "original(1)"]
summary["logit_diff_per_signal_cols_positionsignal"] = ["inverted(-1)", "original(1)"]
summary["logit_diff_per_signal"] = logit_diff_per_signal.tolist()
print("(iii) logit_diff_per_signal (rows tok inv/unc/orig, cols pos inv/orig):", flush=True)
print(logit_diff_per_signal, flush=True)

summary["duration_seconds"] = time.time() - t0
with open(os.path.join(OUT, "step5_sinhibition_result.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("DONE", flush=True)
